"""
Database operations for mystery box allocation.

Adapted from the price scraper's db.py. Supports SSH tunneling for secure
database access through bastion hosts.
"""

import atexit
import functools
import logging
import os
import socket
import threading
from pathlib import Path

import mysql.connector
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SSH tunnel (reused from price scraper)
# ---------------------------------------------------------------------------

class _ParamikoTunnel:
    """SSH tunnel using paramiko directly."""

    def __init__(
        self,
        ssh_host: str,
        ssh_port: int,
        ssh_user: str,
        remote_host: str,
        remote_port: int,
        ssh_password: str | None = None,
        ssh_key_path: str | None = None,
        ssh_key_passphrase: str | None = None,
    ):
        import paramiko

        self._remote_host = remote_host
        self._remote_port = remote_port
        self._shutdown_event = threading.Event()

        self._client = paramiko.SSHClient()
        self._client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        connect_kwargs: dict = {
            "hostname": ssh_host,
            "port": ssh_port,
            "username": ssh_user,
            "look_for_keys": False,
            "allow_agent": False,
        }
        if ssh_password:
            connect_kwargs["password"] = ssh_password
        else:
            connect_kwargs["key_filename"] = ssh_key_path
            if ssh_key_passphrase:
                connect_kwargs["passphrase"] = ssh_key_passphrase

        self._client.connect(**connect_kwargs)
        self._transport = self._client.get_transport()

        self._server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_sock.bind(("127.0.0.1", 0))
        self._server_sock.listen(5)
        self._server_sock.settimeout(1.0)
        self._local_bind_port = self._server_sock.getsockname()[1]

        self._accept_thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._accept_thread.start()

    @property
    def local_bind_port(self) -> int:
        return self._local_bind_port

    def _accept_loop(self):
        while not self._shutdown_event.is_set():
            try:
                client_sock, _ = self._server_sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break

            try:
                channel = self._transport.open_channel(
                    "direct-tcpip",
                    (self._remote_host, self._remote_port),
                    client_sock.getpeername(),
                )
            except Exception as e:
                logger.warning(f"SSH channel open failed: {e}")
                client_sock.close()
                continue

            if channel is None:
                client_sock.close()
                continue

            t1 = threading.Thread(target=self._forward, args=(client_sock, channel), daemon=True)
            t2 = threading.Thread(target=self._forward, args=(channel, client_sock), daemon=True)
            t1.start()
            t2.start()

    def _forward(self, src, dst):
        try:
            src.settimeout(2.0)
        except Exception:
            pass
        try:
            while not self._shutdown_event.is_set():
                try:
                    data = src.recv(4096)
                    if not data:
                        break
                    dst.sendall(data)
                except socket.timeout:
                    continue
                except OSError:
                    break
        finally:
            for s in (src, dst):
                try:
                    s.close()
                except Exception:
                    pass

    def stop(self):
        self._shutdown_event.set()
        try:
            self._server_sock.close()
        except Exception:
            pass
        try:
            self._client.close()
        except Exception:
            pass


class TunnelManager:
    """Singleton manager for SSH tunnel connections with reference counting."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        with TunnelManager._lock:
            if self._initialized:
                return
            self._tunnel = None
            self._ref_count = 0
            self._tunnel_lock = threading.Lock()
            self._initialized = True
            atexit.register(self._cleanup)

    def _cleanup(self):
        with self._tunnel_lock:
            if self._tunnel is not None:
                self._tunnel.stop()
                self._tunnel = None
                self._ref_count = 0

    def acquire(self) -> int:
        with self._tunnel_lock:
            if self._tunnel is None:
                self._start()
            self._ref_count += 1
            return self._tunnel.local_bind_port

    def release(self):
        with self._tunnel_lock:
            self._ref_count = max(0, self._ref_count - 1)
            # Keep tunnel alive - only cleaned up on exit

    def _start(self):
        ssh_host = os.getenv("SSH_HOST")
        ssh_port = int(os.getenv("SSH_PORT", "22"))
        ssh_user = os.getenv("SSH_USER")
        ssh_password = os.getenv("SSH_PASSWORD") or None
        _key_path = os.getenv("SSH_KEY_PATH")
        ssh_key_path = str(Path(_key_path).expanduser()) if _key_path else None
        ssh_key_passphrase = os.getenv("SSH_KEY_PASSPHRASE") or None

        remote_host = os.getenv("SSH_REMOTE_BIND_HOST", os.getenv("DB_HOST", "localhost"))
        remote_port = int(os.getenv("SSH_REMOTE_BIND_PORT", os.getenv("DB_PORT", "3306")))

        logger.info(f"Starting SSH tunnel to {ssh_host}:{ssh_port}")

        self._tunnel = _ParamikoTunnel(
            ssh_host=ssh_host,
            ssh_port=ssh_port,
            ssh_user=ssh_user,
            remote_host=remote_host,
            remote_port=remote_port,
            ssh_password=ssh_password,
            ssh_key_path=ssh_key_path,
            ssh_key_passphrase=ssh_key_passphrase,
        )
        logger.info(f"SSH tunnel on local port {self._tunnel.local_bind_port}")


def _is_ssh_enabled() -> bool:
    return os.getenv("SSH_ENABLED", "false").lower() in ("true", "1", "yes")


def get_connection() -> mysql.connector.MySQLConnection:
    """Create a database connection, optionally through SSH tunnel."""
    if _is_ssh_enabled():
        tunnel_manager = TunnelManager()
        local_port = tunnel_manager.acquire()
        try:
            conn = mysql.connector.connect(
                host="127.0.0.1",
                port=local_port,
                database=os.getenv("DB_NAME", "jointly_db"),
                user=os.getenv("DB_USER"),
                password=os.getenv("DB_PASSWORD"),
            )
        except Exception:
            tunnel_manager.release()
            raise

        original_close = conn.close
        _released = False
        _close_lock = threading.Lock()

        def close_with_release():
            nonlocal _released
            with _close_lock:
                if _released:
                    return
                _released = True
            original_close()
            tunnel_manager.release()

        conn.close = close_with_release
        return conn
    else:
        return mysql.connector.connect(
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", "3306")),
            database=os.getenv("DB_NAME", "jointly_db"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
        )


# ---------------------------------------------------------------------------
# Mystery-box-specific queries
# ---------------------------------------------------------------------------

@functools.cache
def fetch_offer_items(offer_id: int) -> list[dict]:
    """
    Fetch all F&V items for an offer.

    Returns list of dicts with: id, name, price, part_category_id, size,
    pack_order, buy_qty, buy_price, components
    """
    query = """
        SELECT id, name, price, part_category_id, size, pack_order,
               buy_qty, buy_price, components
        FROM offer_parts
        WHERE offer_id = %s
          AND part_category_id IN (2, 3)
          AND deleted_at IS NULL
        ORDER BY pack_order, id
    """
    with get_connection() as conn:
        with conn.cursor(dictionary=True) as cur:
            cur.execute(query, (offer_id,))
            return cur.fetchall()


@functools.cache
def fetch_offer_parts_by_name(offer_id: int) -> dict[str, dict]:
    """
    Fetch F&V offer parts keyed by name for name-based matching (Tier C).

    Returns {name: {id, name, price, part_category_id, pack_order}}.
    """
    query = """
        SELECT id, name, price, part_category_id, pack_order
        FROM offer_parts
        WHERE offer_id = %s
          AND part_category_id IN (2, 3)
          AND deleted_at IS NULL
        ORDER BY pack_order, id
    """
    with get_connection() as conn:
        with conn.cursor(dictionary=True) as cur:
            cur.execute(query, (offer_id,))
            return {row["name"]: row for row in cur.fetchall()}


@functools.cache
def fetch_categories() -> dict[int, str]:
    """Fetch category ID → name mapping."""
    query = "SELECT id, name FROM part_categories"
    with get_connection() as conn:
        with conn.cursor(dictionary=True) as cur:
            cur.execute(query)
            return {row["id"]: row["name"] for row in cur.fetchall()}


@functools.cache
def fetch_mystery_box_buyers(offer_id: int) -> list[dict]:
    """
    Find users who bought mystery boxes for this offer.

    Returns list of dicts with: user_id, user_email, offer_part_id,
    offer_part_name, buy_id, qty, buy_options_json, selected_option,
    note_to_seller
    """
    query = """
        SELECT
            u.id AS user_id,
            u.email AS user_email,
            op.id AS offer_part_id,
            op.name AS offer_part_name,
            op.buy_options AS buy_options_json,
            b.id AS buy_id,
            bp.qty,
            bp.options AS selected_option,
            b.note_to_seller
        FROM buy_parts bp
        JOIN buys b ON b.id = bp.buy_id
        JOIN users u ON u.id = b.user_id
        JOIN offer_parts op ON op.id = bp.offer_part_id
        WHERE b.offer_id = %s
          AND b.deleted_at IS NULL
          AND bp.deleted_at IS NULL
          AND op.deleted_at IS NULL
          AND LOWER(op.name) LIKE '%%mystery box%%'
        ORDER BY op.name, u.email
    """
    with get_connection() as conn:
        with conn.cursor(dictionary=True) as cur:
            cur.execute(query, (offer_id,))
            return cur.fetchall()


@functools.cache
def fetch_buyer_existing_categories(offer_id: int, user_email: str) -> set[int]:
    """
    Get category IDs of items a buyer already has in their regular order.

    Used for merged boxes to avoid duplicating categories.
    """
    query = """
        SELECT DISTINCT op.part_category_id
        FROM buy_parts bp
        JOIN buys b ON b.id = bp.buy_id
        JOIN users u ON u.id = b.user_id
        JOIN offer_parts op ON op.id = bp.offer_part_id
        WHERE b.offer_id = %s
          AND u.email = %s
          AND bp.qty > 0
          AND b.deleted_at IS NULL
          AND bp.deleted_at IS NULL
          AND op.deleted_at IS NULL
          AND LOWER(op.name) NOT LIKE '%%mystery box%%'
    """
    with get_connection() as conn:
        with conn.cursor(dictionary=True) as cur:
            cur.execute(query, (offer_id, user_email))
            return {row["part_category_id"] for row in cur.fetchall()}


@functools.cache
def fetch_offer_gross_retail(offer_id: int) -> int:
    """
    Calculate gross retail value (sum of price * qty sold) for an offer.
    Returns value in cents.
    """
    query = """
        SELECT COALESCE(SUM(op.price * bp.qty), 0) AS gross
        FROM buy_parts bp
        JOIN buys b ON b.id = bp.buy_id
        JOIN offer_parts op ON op.id = bp.offer_part_id
        WHERE b.offer_id = %s
          AND b.deleted_at IS NULL
          AND bp.deleted_at IS NULL
          AND op.deleted_at IS NULL
    """
    with get_connection() as conn:
        with conn.cursor(dictionary=True) as cur:
            cur.execute(query, (offer_id,))
            row = cur.fetchone()
            return row["gross"] or 0


@functools.cache
def fetch_customer_giving(offer_id: int) -> int:
    """
    Calculate total customer giving/donations for the offer.
    Returns value in cents. Giving comes from buys where offer_part
    is in the 'Giving' category (part_category_id = 6).
    """
    query = """
        SELECT COALESCE(SUM(op.price * bp.qty), 0) AS giving
        FROM buy_parts bp
        JOIN buys b ON b.id = bp.buy_id
        JOIN offer_parts op ON op.id = bp.offer_part_id
        WHERE b.offer_id = %s
          AND op.part_category_id = 6
          AND b.deleted_at IS NULL
          AND bp.deleted_at IS NULL
          AND op.deleted_at IS NULL
    """
    with get_connection() as conn:
        with conn.cursor(dictionary=True) as cur:
            cur.execute(query, (offer_id,))
            row = cur.fetchone()
            return row["giving"] or 0


@functools.cache
def fetch_recent_offer_ids(limit: int = 10) -> list[int]:
    """Fetch recent offer IDs that have F&V items."""
    query = """
        SELECT DISTINCT offer_id
        FROM offer_parts
        WHERE part_category_id IN (2, 3)
          AND deleted_at IS NULL
        ORDER BY offer_id DESC
        LIMIT %s
    """
    with get_connection() as conn:
        with conn.cursor(dictionary=True) as cur:
            cur.execute(query, (limit,))
            return [row["offer_id"] for row in cur.fetchall()]
