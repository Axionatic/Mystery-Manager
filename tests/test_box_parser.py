"""Tests for allocator/box_parser.py — parse_box_name and classify_box."""

import pytest

from allocator.box_parser import classify_box, parse_box_name


# ── parse_box_name ──────────────────────────────────────────────────────────


class TestParseBoxName:
    # -- Question mark prefix --

    def test_question_sm_name(self):
        name, tier = parse_box_name("?Sm Alex")
        assert tier == "small"
        assert name == "Alex"

    def test_question_md_name(self):
        name, tier = parse_box_name("?Md Bob")
        assert tier == "medium"

    def test_question_lg_name(self):
        name, tier = parse_box_name("?Lg Charlie")
        assert tier == "large"

    def test_paren_question_prefix(self):
        name, tier = parse_box_name("(?) Lg Dave")
        assert tier == "large"
        assert name == "Dave"

    # -- Size prefix --

    def test_sm_prefix(self):
        name, tier = parse_box_name("Sm Eve")
        assert tier == "small"
        assert name == "Eve"

    def test_sml_prefix(self):
        name, tier = parse_box_name("Sml Frank")
        assert tier == "small"
        assert name == "Frank"

    def test_small_prefix(self):
        name, tier = parse_box_name("Small Grace")
        assert tier == "small"

    def test_md_prefix(self):
        name, tier = parse_box_name("Md Hank")
        assert tier == "medium"

    def test_med_prefix(self):
        name, tier = parse_box_name("Med Ivy")
        assert tier == "medium"

    def test_lg_prefix(self):
        name, tier = parse_box_name("Lg Jack")
        assert tier == "large"

    def test_lge_prefix(self):
        name, tier = parse_box_name("Lge Kate")
        assert tier == "large"

    # -- Size - Name pattern --

    def test_size_dash_name(self):
        name, tier = parse_box_name("Small - Mark")
        assert tier == "small"
        assert name == "Mark"

    def test_medium_dash_name(self):
        name, tier = parse_box_name("Medium - Nora")
        assert tier == "medium"
        assert name == "Nora"

    # -- Letter Box N pattern --

    def test_m_box_n(self):
        name, tier = parse_box_name("M Box 1")
        assert tier == "medium"
        assert name == "M Box 1"

    def test_l_box_n(self):
        name, tier = parse_box_name("L Box 2")
        assert tier == "large"

    def test_s_box_n(self):
        name, tier = parse_box_name("S Box 3")
        assert tier == "small"

    # -- Ordinals --

    def test_ordinal_2nd_md(self):
        name, tier = parse_box_name("2nd Md")
        assert tier == "medium"
        assert name == "2nd Md"

    def test_ordinal_4th_medium(self):
        name, tier = parse_box_name("4th Medium")
        assert tier == "medium"

    # -- Plus suffix --

    def test_md_plus(self):
        name, tier = parse_box_name("Md+")
        assert tier == "medium"

    def test_sm_plus(self):
        name, tier = parse_box_name("Sm+")
        assert tier == "small"

    # -- Number suffix --

    def test_small_number(self):
        name, tier = parse_box_name("Small1")
        assert tier == "small"

    def test_med_number(self):
        name, tier = parse_box_name("Med1")
        assert tier == "medium"

    def test_sm_number(self):
        name, tier = parse_box_name("SM 1")
        assert tier == "small"

    # -- Box NN prefix --

    def test_box_num_prefix(self):
        name, tier = parse_box_name("Box 26: Lge Charity")
        assert tier == "large"

    def test_box_num_md(self):
        name, tier = parse_box_name("Box 27: Md Name #1")
        assert tier == "medium"

    # -- Market / mystery --

    def test_market_defaults_small(self):
        name, tier = parse_box_name("Market Box")
        assert tier == "small"

    def test_mystery_defaults_small(self):
        name, tier = parse_box_name("Mystery Item")
        assert tier == "small"

    def test_4sale_defaults_small(self):
        name, tier = parse_box_name("Sm 4Sale")
        assert tier == "small"

    # -- Edge cases --

    def test_empty_string(self):
        name, tier = parse_box_name("")
        assert name == ""
        assert tier is None

    def test_no_size_indicator(self):
        name, tier = parse_box_name("Random Name")
        assert tier is None

    def test_email_address_no_size(self):
        name, tier = parse_box_name("customer@example.com")
        assert tier is None

    def test_whitespace_stripping(self):
        name, tier = parse_box_name("  Sm Alex  ")
        assert tier == "small"
        assert name == "Alex"


# ── classify_box ────────────────────────────────────────────────────────────


class TestClassifyBox:
    def test_email_is_merged(self, monkeypatch):
        # Ensure this email is not in donation or staff identifiers
        monkeypatch.setattr("allocator.box_parser.DONATION_IDENTIFIERS", set())
        monkeypatch.setattr("allocator.box_parser.STAFF_IDENTIFIERS", set())
        _, _, box_type = classify_box("customer@example.com")
        assert box_type == "merged"

    def test_standalone_name(self, monkeypatch):
        monkeypatch.setattr("allocator.box_parser.STANDALONE_NAME_TO_EMAIL", {})
        monkeypatch.setattr("allocator.box_parser.DONATION_IDENTIFIERS", set())
        monkeypatch.setattr("allocator.box_parser.STAFF_IDENTIFIERS", set())
        monkeypatch.setattr("allocator.box_parser._UNRELIABLE_SIZE_NAMES", set())
        _, _, box_type = classify_box("Sm Alex")
        assert box_type == "standalone"

    def test_donation_email(self, monkeypatch):
        monkeypatch.setattr("allocator.box_parser.DONATION_IDENTIFIERS", {"donor@test.example"})
        monkeypatch.setattr("allocator.box_parser.STAFF_IDENTIFIERS", set())
        _, _, box_type = classify_box("donor@test.example")
        assert box_type == "donation"

    def test_staff_email(self, monkeypatch):
        monkeypatch.setattr("allocator.box_parser.DONATION_IDENTIFIERS", set())
        monkeypatch.setattr("allocator.box_parser.STAFF_IDENTIFIERS", {"staff@test.example"})
        _, _, box_type = classify_box("staff@test.example")
        assert box_type == "staff"

    def test_charity_name(self, monkeypatch):
        monkeypatch.setattr("allocator.box_parser.CHARITY_NAME", "Test Charity")
        monkeypatch.setattr("allocator.box_parser.DONATION_IDENTIFIERS", set())
        monkeypatch.setattr("allocator.box_parser.STAFF_IDENTIFIERS", set())
        monkeypatch.setattr("allocator.box_parser.STANDALONE_NAME_TO_EMAIL", {})
        monkeypatch.setattr("allocator.box_parser._UNRELIABLE_SIZE_NAMES", set())
        _, _, box_type = classify_box("Sm Test Charity")
        assert box_type == "donation"

    def test_standalone_alias_is_merged(self, monkeypatch):
        monkeypatch.setattr("allocator.box_parser.STANDALONE_NAME_TO_EMAIL", {"TestAlias": "alias@test.example"})
        monkeypatch.setattr("allocator.box_parser.DONATION_IDENTIFIERS", set())
        monkeypatch.setattr("allocator.box_parser.STAFF_IDENTIFIERS", set())
        monkeypatch.setattr("allocator.box_parser._UNRELIABLE_SIZE_NAMES", set())
        _, _, box_type = classify_box("Sm TestAlias")
        assert box_type == "merged"

    def test_empty_is_skip(self):
        _, _, box_type = classify_box("")
        assert box_type == "skip"

    def test_none_is_skip(self):
        _, _, box_type = classify_box(None)
        assert box_type == "skip"
