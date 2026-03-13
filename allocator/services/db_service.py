"""DB connectivity service -- thin wrapper around allocator.db."""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class DBService:
    """Provides database connectivity checks for the TUI.

    Wraps allocator.db.get_connection() without reimplementing any
    connection or tunnel logic. Concrete class -- no ABC/protocols
    until a web UI is actually built.
    """

    def check_connectivity(self, timeout: int = 5) -> bool:
        """Attempt a DB connection; return True if successful.

        Calls db.get_connection() with a timeout guard, executes
        SELECT 1 to verify the connection is live, then closes
        immediately. Returns False on any exception (connection
        refused, SSH tunnel failure, missing .env vars, timeout,
        etc.).

        This method is synchronous and blocking. Callers must run
        it in a thread worker (e.g. Textual @work(thread=True)) to
        avoid blocking the event loop.
        """
        from concurrent.futures import ThreadPoolExecutor, TimeoutError

        def _ping():
            from dotenv import load_dotenv
            load_dotenv()

            import os
            import mysql.connector

            kwargs = {
                "database": os.getenv("DB_NAME", "app_db"),
                "user": os.getenv("DB_USER"),
                "password": os.getenv("DB_PASSWORD"),
                "connection_timeout": 5,
            }
            unix_socket = os.getenv("DB_SOCKET")
            if unix_socket:
                kwargs["unix_socket"] = unix_socket
            else:
                kwargs["host"] = os.getenv("DB_HOST", "localhost")
                kwargs["port"] = int(os.getenv("DB_PORT", "3306"))

            ssh_enabled = os.getenv("SSH_ENABLED", "false").lower() == "true"
            if ssh_enabled:
                # For SSH tunnel connections, use get_connection() which manages the tunnel
                from allocator import db as _db
                conn = _db.get_connection()
            else:
                conn = mysql.connector.connect(**kwargs)

            try:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    cur.fetchone()
            finally:
                conn.close()

        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                pool.submit(_ping).result(timeout=timeout)
            logger.debug("DB connectivity check passed")
            return True
        except TimeoutError:
            logger.warning("DB connectivity check timed out after %ds", timeout)
            return False
        except Exception as exc:
            logger.warning("DB connectivity check failed: %s", exc)
            return False
