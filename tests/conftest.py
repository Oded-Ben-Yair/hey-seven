"""Shared test fixtures for Hey Seven tests."""

import json

import pytest


@pytest.fixture
def test_property_data():
    """Small test dataset for property Q&A tests."""
    return {
        "property": {"name": "Test Casino", "location": "Test City, NV"},
        "restaurants": [
            {
                "name": "Test Steakhouse",
                "cuisine": "Steakhouse",
                "price_range": "$$$",
                "hours": "5-10 PM",
                "location": "Main Floor",
                "description": "A fine steakhouse",
            },
            {
                "name": "Test Buffet",
                "cuisine": "Buffet",
                "price_range": "$$",
                "hours": "11 AM-9 PM",
                "location": "Level 2",
                "description": "All you can eat",
            },
        ],
        "entertainment": [
            {
                "name": "Test Arena",
                "type": "Concert Venue",
                "capacity": "10000",
                "description": "Main arena",
            }
        ],
        "hotel": {
            "towers": [
                {
                    "name": "Sky Tower",
                    "description": "Luxury tower with mountain views",
                    "floors": 34,
                }
            ],
            "room_types": [
                {
                    "name": "Deluxe King",
                    "size": "400 sq ft",
                    "rate": "$199/night",
                    "description": "Spacious room with king bed",
                }
            ],
        },
        "gaming": {
            "casino_size_sqft": 300000,
            "slot_machines": 5000,
            "table_games": 300,
        },
        "faq": [{"question": "What are the hours?", "answer": "Open 24/7"}],
    }


@pytest.fixture
def test_property_file(tmp_path, test_property_data):
    """Write test data to a temp JSON file and return its path."""
    p = tmp_path / "test_property.json"
    p.write_text(json.dumps(test_property_data))
    return str(p)
