"""Build a dataset of beauty foundation products across brands.

Produces `foundations.csv` with one row per (product, shade) combination.

Data caveats:
- Prices are approximate MSRP in USD at time of authoring and may drift.
- Shade codes reflect each brand's real naming convention where known; a
  handful of brands use only shade names (no separate code) and those cells
  are left blank rather than fabricated.
- Depth and undertone buckets are normalized across brands using a common
  vocabulary (see README).
- Ingredient notes summarize marketed hero ingredients / formula family,
  not the full INCI list.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Controlled vocabularies
# ---------------------------------------------------------------------------

DEPTH_BUCKETS = [
    "Fair",
    "Light",
    "Light-Medium",
    "Medium",
    "Medium-Tan",
    "Tan",
    "Deep",
    "Rich",
]

UNDERTONE_BUCKETS = ["Cool", "Neutral", "Warm", "Olive", "Neutral-Cool", "Neutral-Warm"]

FINISHES = ["Matte", "Natural", "Satin", "Radiant", "Dewy", "Luminous"]

COVERAGES = ["Sheer", "Light", "Light-Medium", "Medium", "Medium-Full", "Full"]


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Shade:
    name: str
    code: str
    depth: str
    undertone: str


@dataclass
class Product:
    brand: str
    product: str
    finish: str
    coverage: str
    skin_types: str            # e.g. "Oily, Combination"
    spf: int                   # 0 if none
    fragrance_free: bool
    price_usd: float
    ingredient_notes: str
    shades: list[Shade] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Products
# Each product contains a representative sample of shades (3-6 typically)
# spanning the depth spectrum offered by that line.
# ---------------------------------------------------------------------------

PRODUCTS: list[Product] = [

    # ---- Fenty Beauty ----
    Product(
        "Fenty Beauty", "Pro Filt'r Soft Matte Longwear Foundation",
        finish="Matte", coverage="Medium-Full",
        skin_types="Oily, Combination, Normal", spf=0, fragrance_free=True,
        price_usd=42.0,
        ingredient_notes="Water-based, climate-adaptive; grape seed extract; long-wear polymers.",
        shades=[
            Shade("110", "110", "Fair", "Cool"),
            Shade("150", "150", "Light", "Neutral"),
            Shade("240", "240", "Light-Medium", "Warm"),
            Shade("330", "330", "Medium", "Olive"),
            Shade("410", "410", "Tan", "Warm"),
            Shade("480", "480", "Deep", "Neutral"),
            Shade("498", "498", "Rich", "Cool"),
        ],
    ),
    Product(
        "Fenty Beauty", "Eaze Drop Blurring Skin Tint",
        finish="Natural", coverage="Sheer",
        skin_types="All", spf=0, fragrance_free=True,
        price_usd=32.0,
        ingredient_notes="Water-gel; blurring powders; kalahari melon; lightweight silicones.",
        shades=[
            Shade("2", "2", "Fair", "Neutral"),
            Shade("6", "6", "Light-Medium", "Warm"),
            Shade("11", "11", "Medium", "Neutral"),
            Shade("17", "17", "Tan", "Warm"),
            Shade("22", "22", "Deep", "Neutral"),
        ],
    ),

    # ---- MAC ----
    Product(
        "MAC", "Studio Fix Fluid SPF 15",
        finish="Matte", coverage="Medium-Full",
        skin_types="Oily, Combination", spf=15, fragrance_free=False,
        price_usd=36.0,
        ingredient_notes="Silica; talc; titanium/zinc SPF filters; oil-controlling powders.",
        shades=[
            Shade("NW10", "NW10", "Fair", "Cool"),
            Shade("NC15", "NC15", "Light", "Warm"),
            Shade("NC25", "NC25", "Light-Medium", "Warm"),
            Shade("NW35", "NW35", "Medium", "Neutral-Warm"),
            Shade("NC45", "NC45", "Tan", "Warm"),
            Shade("NW50", "NW50", "Deep", "Warm"),
            Shade("NW58", "NW58", "Rich", "Warm"),
        ],
    ),
    Product(
        "MAC", "Studio Radiance Serum-Powered Foundation",
        finish="Radiant", coverage="Light-Medium",
        skin_types="Dry, Normal, Combination", spf=0, fragrance_free=False,
        price_usd=42.0,
        ingredient_notes="Hyaluronic acid; glycerin; skincare-serum base.",
        shades=[
            Shade("NC15", "NC15", "Light", "Warm"),
            Shade("NW25", "NW25", "Light-Medium", "Neutral-Warm"),
            Shade("NC42", "NC42", "Medium-Tan", "Warm"),
            Shade("NW47", "NW47", "Tan", "Warm"),
            Shade("NW58", "NW58", "Deep", "Warm"),
        ],
    ),

    # ---- NARS ----
    Product(
        "NARS", "Natural Radiant Longwear Foundation",
        finish="Radiant", coverage="Medium",
        skin_types="Normal, Combination, Dry", spf=0, fragrance_free=False,
        price_usd=52.0,
        ingredient_notes="Skin-tone optimizing complex; hyaluronic acid; 16-hr wear.",
        shades=[
            Shade("Mont Blanc", "MB", "Fair", "Neutral-Cool"),
            Shade("Deauville", "DV", "Light", "Neutral"),
            Shade("Punjab", "PJ", "Light-Medium", "Warm"),
            Shade("Barcelona", "BC", "Medium", "Warm"),
            Shade("Syracuse", "SY", "Tan", "Warm"),
            Shade("Macao", "MC", "Deep", "Neutral"),
            Shade("Marquises", "MQ", "Rich", "Warm"),
        ],
    ),
    Product(
        "NARS", "Sheer Glow Foundation",
        finish="Radiant", coverage="Light-Medium",
        skin_types="Normal, Dry, Combination", spf=0, fragrance_free=False,
        price_usd=52.0,
        ingredient_notes="Vitamin C; turmeric extract; buildable water-in-oil emulsion.",
        shades=[
            Shade("Siberia", "SI", "Fair", "Cool"),
            Shade("Gobi", "GB", "Light", "Neutral"),
            Shade("Fiji", "FJ", "Light-Medium", "Warm"),
            Shade("Barcelona", "BC", "Medium", "Warm"),
            Shade("New Orleans", "NO", "Medium-Tan", "Neutral"),
            Shade("Trinidad", "TR", "Deep", "Warm"),
        ],
    ),
    Product(
        "NARS", "Soft Matte Complete Foundation",
        finish="Matte", coverage="Full",
        skin_types="Oily, Combination", spf=0, fragrance_free=False,
        price_usd=55.0,
        ingredient_notes="Full-coverage cream-powder hybrid; oil-absorbing polymers.",
        shades=[
            Shade("Mont Blanc", "MB", "Fair", "Neutral-Cool"),
            Shade("Vallauris", "VL", "Light", "Neutral"),
            Shade("Punjab", "PJ", "Light-Medium", "Warm"),
            Shade("Cadiz", "CD", "Medium-Tan", "Warm"),
            Shade("Macao", "MC", "Deep", "Neutral"),
        ],
    ),

    # ---- Estée Lauder ----
    Product(
        "Estée Lauder", "Double Wear Stay-in-Place Foundation SPF 10",
        finish="Natural", coverage="Full",
        skin_types="Oily, Combination, Normal", spf=10, fragrance_free=True,
        price_usd=52.0,
        ingredient_notes="24-hr transfer-resistant; silicone film-formers; titanium dioxide SPF.",
        shades=[
            Shade("Fresco 2C3", "2C3", "Light", "Cool"),
            Shade("Ivory Nude 1N1", "1N1", "Fair", "Neutral"),
            Shade("Desert Beige 2N1", "2N1", "Light-Medium", "Neutral"),
            Shade("Rich Cocoa 6C1", "6C1", "Deep", "Cool"),
            Shade("Tawny 3W1", "3W1", "Medium", "Warm"),
            Shade("Amber Honey 5W1", "5W1", "Tan", "Warm"),
            Shade("Espresso 7N1", "7N1", "Rich", "Neutral"),
        ],
    ),

    # ---- Maybelline ----
    Product(
        "Maybelline", "Fit Me Matte + Poreless Foundation",
        finish="Matte", coverage="Medium",
        skin_types="Oily, Combination", spf=0, fragrance_free=False,
        price_usd=8.99,
        ingredient_notes="Micro-powders; blurring pigments; drugstore staple.",
        shades=[
            Shade("Porcelain", "110", "Fair", "Cool"),
            Shade("Classic Ivory", "120", "Light", "Neutral"),
            Shade("Natural Beige", "220", "Light-Medium", "Warm"),
            Shade("Toffee", "330", "Medium-Tan", "Warm"),
            Shade("Mocha", "355", "Deep", "Warm"),
            Shade("Espresso", "375", "Rich", "Warm"),
        ],
    ),
    Product(
        "Maybelline", "Fit Me Dewy + Smooth Foundation SPF 18",
        finish="Dewy", coverage="Light-Medium",
        skin_types="Dry, Normal", spf=18, fragrance_free=False,
        price_usd=8.99,
        ingredient_notes="Water-based; glycerin; chemical SPF filters.",
        shades=[
            Shade("Ivory", "115", "Fair", "Neutral"),
            Shade("Classic Beige", "230", "Light-Medium", "Neutral"),
            Shade("Warm Sun", "310", "Medium", "Warm"),
            Shade("Coconut", "355", "Deep", "Warm"),
        ],
    ),
    Product(
        "Maybelline", "Super Stay 24H Full Coverage Foundation",
        finish="Matte", coverage="Full",
        skin_types="Oily, Combination, Normal", spf=0, fragrance_free=False,
        price_usd=12.99,
        ingredient_notes="Micro-flex technology; 24-hr wear; transfer-resistant film.",
        shades=[
            Shade("Ivory", "112", "Fair", "Neutral"),
            Shade("Natural Beige", "220", "Light-Medium", "Warm"),
            Shade("Truffle", "362", "Medium-Tan", "Neutral"),
            Shade("Mocha", "368", "Deep", "Warm"),
        ],
    ),

    # ---- L'Oréal Paris ----
    Product(
        "L'Oréal Paris", "True Match Super-Blendable Foundation SPF 17",
        finish="Natural", coverage="Medium",
        skin_types="Normal, Combination, Dry", spf=17, fragrance_free=False,
        price_usd=12.99,
        ingredient_notes="Precise Match Technology; chemical SPF; grapeseed oil.",
        shades=[
            Shade("Porcelain", "N1", "Fair", "Neutral"),
            Shade("Light Ivory", "W2", "Light", "Warm"),
            Shade("Sand Beige", "N4", "Light-Medium", "Neutral"),
            Shade("Caramel Beige", "W6", "Medium-Tan", "Warm"),
            Shade("Cocoa", "N9", "Deep", "Neutral"),
            Shade("Ebony", "C10", "Rich", "Cool"),
        ],
    ),
    Product(
        "L'Oréal Paris", "Infallible 24H Fresh Wear Foundation",
        finish="Natural", coverage="Full",
        skin_types="Normal, Combination, Oily", spf=0, fragrance_free=False,
        price_usd=15.99,
        ingredient_notes="Breathable film-formers; transfer-resistant; up to 24-hr wear.",
        shades=[
            Shade("Porcelain", "100", "Fair", "Neutral"),
            Shade("Ivory Buff", "125", "Light", "Warm"),
            Shade("True Beige", "200", "Light-Medium", "Neutral"),
            Shade("Deep Amber", "410", "Tan", "Warm"),
            Shade("Espresso", "525", "Deep", "Neutral"),
        ],
    ),

    # ---- Rare Beauty ----
    Product(
        "Rare Beauty", "Liquid Touch Weightless Foundation",
        finish="Natural", coverage="Light-Medium",
        skin_types="All", spf=0, fragrance_free=True,
        price_usd=31.0,
        ingredient_notes="Lotus, gardenia & water-lily complex; buildable; skincare-first.",
        shades=[
            Shade("110C", "110C", "Fair", "Cool"),
            Shade("140N", "140N", "Light", "Neutral"),
            Shade("230W", "230W", "Light-Medium", "Warm"),
            Shade("320N", "320N", "Medium", "Neutral"),
            Shade("400W", "400W", "Tan", "Warm"),
            Shade("480N", "480N", "Deep", "Neutral"),
            Shade("540W", "540W", "Rich", "Warm"),
        ],
    ),

    # ---- Charlotte Tilbury ----
    Product(
        "Charlotte Tilbury", "Airbrush Flawless Foundation",
        finish="Matte", coverage="Full",
        skin_types="Combination, Oily, Normal", spf=0, fragrance_free=False,
        price_usd=49.0,
        ingredient_notes="Blur-expanding pigments; hyaluronic acid; replenium; 24-hr wear.",
        shades=[
            Shade("2 Cool", "2C", "Fair", "Cool"),
            Shade("4 Neutral", "4N", "Light", "Neutral"),
            Shade("7 Warm", "7W", "Light-Medium", "Warm"),
            Shade("10 Neutral", "10N", "Medium", "Neutral"),
            Shade("13 Warm", "13W", "Tan", "Warm"),
            Shade("15 Cool", "15C", "Deep", "Cool"),
        ],
    ),
    Product(
        "Charlotte Tilbury", "Beautiful Skin Foundation",
        finish="Radiant", coverage="Light-Medium",
        skin_types="Dry, Normal, Combination", spf=20, fragrance_free=False,
        price_usd=49.0,
        ingredient_notes="Hyaluronic acid; bamboo extract; polyglutamic acid.",
        shades=[
            Shade("1 Fair Cool", "1CF", "Fair", "Cool"),
            Shade("5 Neutral", "5N", "Light-Medium", "Neutral"),
            Shade("9 Warm", "9W", "Medium", "Warm"),
            Shade("14 Neutral", "14N", "Tan", "Neutral"),
            Shade("17 Warm", "17W", "Deep", "Warm"),
        ],
    ),

    # ---- Dior ----
    Product(
        "Dior", "Dior Forever 24H Wear High Perfection SPF 20",
        finish="Matte", coverage="Full",
        skin_types="Oily, Combination", spf=20, fragrance_free=False,
        price_usd=62.0,
        ingredient_notes="Floral skincare complex; up to 24-hr; matte SPF filters.",
        shades=[
            Shade("0N Neutral", "0N", "Fair", "Neutral"),
            Shade("2W Warm", "2W", "Light", "Warm"),
            Shade("3CR Cool Rosy", "3CR", "Light-Medium", "Cool"),
            Shade("4WO Warm Olive", "4WO", "Medium", "Olive"),
            Shade("5N Neutral", "5N", "Medium-Tan", "Neutral"),
            Shade("7N Neutral", "7N", "Deep", "Neutral"),
            Shade("9N Neutral", "9N", "Rich", "Neutral"),
        ],
    ),
    Product(
        "Dior", "Dior Forever Skin Glow SPF 20",
        finish="Radiant", coverage="Medium-Full",
        skin_types="Normal, Dry, Combination", spf=20, fragrance_free=False,
        price_usd=62.0,
        ingredient_notes="Floral skincare; radiance boosters; hydration up to 24-hr.",
        shades=[
            Shade("1N Neutral", "1N", "Fair", "Neutral"),
            Shade("3N Neutral", "3N", "Light-Medium", "Neutral"),
            Shade("4WP Warm Peach", "4WP", "Medium", "Warm"),
            Shade("6N Neutral", "6N", "Tan", "Neutral"),
            Shade("8N Neutral", "8N", "Deep", "Neutral"),
        ],
    ),

    # ---- YSL ----
    Product(
        "YSL Beauty", "All Hours Precise Angles Longwear Foundation",
        finish="Matte", coverage="Full",
        skin_types="Oily, Combination, Normal", spf=30, fragrance_free=False,
        price_usd=55.0,
        ingredient_notes="Hyaluronic acid; 24-hr matte; broad-spectrum SPF 30.",
        shades=[
            Shade("LN1", "LN1", "Fair", "Neutral"),
            Shade("LC2", "LC2", "Light", "Cool"),
            Shade("B40", "B40", "Light-Medium", "Warm"),
            Shade("MN7", "MN7", "Medium-Tan", "Neutral"),
            Shade("DN2", "DN2", "Deep", "Neutral"),
            Shade("DW5", "DW5", "Rich", "Warm"),
        ],
    ),
    Product(
        "YSL Beauty", "Touche Éclat Le Teint Foundation SPF 22",
        finish="Radiant", coverage="Medium",
        skin_types="Normal, Dry, Combination", spf=22, fragrance_free=False,
        price_usd=64.0,
        ingredient_notes="Light-reflecting pigments; hyaluronic acid; radiance boosters.",
        shades=[
            Shade("BR20", "BR20", "Fair", "Cool"),
            Shade("B40", "B40", "Light-Medium", "Neutral-Warm"),
            Shade("BR50", "BR50", "Medium", "Warm"),
            Shade("BR70", "BR70", "Tan", "Warm"),
            Shade("BR80", "BR80", "Deep", "Warm"),
        ],
    ),

    # ---- Giorgio Armani ----
    Product(
        "Giorgio Armani", "Luminous Silk Perfect Glow Flawless Foundation",
        finish="Luminous", coverage="Medium",
        skin_types="Normal, Dry, Combination", spf=0, fragrance_free=False,
        price_usd=69.0,
        ingredient_notes="Micro-fil technology; glycerin; buildable second-skin finish.",
        shades=[
            Shade("2", "2", "Fair", "Cool"),
            Shade("4.5", "4.5", "Light", "Neutral"),
            Shade("6.5", "6.5", "Light-Medium", "Warm"),
            Shade("8", "8", "Medium", "Warm"),
            Shade("10", "10", "Tan", "Neutral"),
            Shade("14", "14", "Deep", "Neutral"),
            Shade("15.5", "15.5", "Rich", "Warm"),
        ],
    ),
    Product(
        "Giorgio Armani", "Power Fabric+ Longwear High Coverage Foundation SPF 25",
        finish="Matte", coverage="Full",
        skin_types="Oily, Combination", spf=25, fragrance_free=False,
        price_usd=69.0,
        ingredient_notes="High-coverage pigments; 16-hr wear; oil-controlling.",
        shades=[
            Shade("2", "2", "Fair", "Cool"),
            Shade("4", "4", "Light", "Neutral"),
            Shade("6", "6", "Light-Medium", "Warm"),
            Shade("9", "9", "Medium-Tan", "Warm"),
            Shade("12", "12", "Deep", "Neutral"),
        ],
    ),

    # ---- Chanel ----
    Product(
        "Chanel", "Les Beiges Water-Fresh Tint",
        finish="Dewy", coverage="Sheer",
        skin_types="All", spf=0, fragrance_free=False,
        price_usd=75.0,
        ingredient_notes="Water-based micro-droplets; rose extract; buildable tint.",
        shades=[
            Shade("Light Deep", "LD", "Fair", "Neutral"),
            Shade("Light", "L", "Light", "Neutral"),
            Shade("Medium Light", "ML", "Light-Medium", "Warm"),
            Shade("Medium Plus", "MP", "Medium", "Neutral"),
            Shade("Deep", "D", "Tan", "Warm"),
            Shade("Deep Plus", "DP", "Deep", "Warm"),
        ],
    ),
    Product(
        "Chanel", "Ultra Le Teint Velvet Blurring Smooth-Effect Foundation",
        finish="Matte", coverage="Medium-Full",
        skin_types="Normal, Combination, Oily", spf=15, fragrance_free=False,
        price_usd=68.0,
        ingredient_notes="Blurring powders; 24-hr comfort matte.",
        shades=[
            Shade("BR12", "BR12", "Fair", "Cool"),
            Shade("B30", "B30", "Light", "Warm"),
            Shade("BR42", "BR42", "Light-Medium", "Warm"),
            Shade("B70", "B70", "Medium-Tan", "Warm"),
            Shade("BR152", "BR152", "Deep", "Warm"),
        ],
    ),

    # ---- Clinique ----
    Product(
        "Clinique", "Even Better Makeup SPF 15",
        finish="Natural", coverage="Medium-Full",
        skin_types="Normal, Combination, Oily", spf=15, fragrance_free=True,
        price_usd=36.0,
        ingredient_notes="Vitamin C; salicylic acid; brightening formula.",
        shades=[
            Shade("Ivory", "CN 28", "Fair", "Neutral"),
            Shade("Vanilla", "CN 52", "Light-Medium", "Neutral"),
            Shade("Sand", "WN 76", "Medium", "Warm"),
            Shade("Amber", "WN 114", "Tan", "Warm"),
            Shade("Espresso", "WN 125", "Deep", "Warm"),
        ],
    ),
    Product(
        "Clinique", "Beyond Perfecting Foundation + Concealer",
        finish="Natural", coverage="Full",
        skin_types="Normal, Combination, Dry", spf=0, fragrance_free=True,
        price_usd=36.0,
        ingredient_notes="Blends concealer & foundation; 24-hr moisture-lock; hypoallergenic.",
        shades=[
            Shade("Alabaster", "02", "Fair", "Cool"),
            Shade("Ivory", "06", "Light", "Neutral"),
            Shade("Vanilla", "09", "Light-Medium", "Warm"),
            Shade("Sand", "14", "Medium", "Warm"),
            Shade("Golden", "18", "Tan", "Warm"),
        ],
    ),

    # ---- Bobbi Brown ----
    Product(
        "Bobbi Brown", "Skin Long-Wear Weightless Foundation SPF 15",
        finish="Natural", coverage="Medium-Full",
        skin_types="Normal, Combination, Oily", spf=15, fragrance_free=True,
        price_usd=52.0,
        ingredient_notes="Oil-free; hydration + oil control; 16-hr wear.",
        shades=[
            Shade("Alabaster", "0", "Fair", "Cool"),
            Shade("Warm Ivory", "1", "Light", "Warm"),
            Shade("Beige", "3", "Light-Medium", "Neutral"),
            Shade("Natural Tan", "4.5", "Medium-Tan", "Warm"),
            Shade("Warm Almond", "5.75", "Deep", "Warm"),
            Shade("Cool Espresso", "8", "Rich", "Cool"),
        ],
    ),
    Product(
        "Bobbi Brown", "Intensive Skin Serum Foundation SPF 40",
        finish="Radiant", coverage="Medium",
        skin_types="Dry, Normal, Mature", spf=40, fragrance_free=True,
        price_usd=79.0,
        ingredient_notes="Cordyceps mushroom; watermelon; broad-spectrum SPF 40 serum-foundation hybrid.",
        shades=[
            Shade("Porcelain", "N-012", "Fair", "Neutral"),
            Shade("Natural", "N-052", "Light-Medium", "Neutral"),
            Shade("Warm Beige", "W-046", "Medium", "Warm"),
            Shade("Warm Almond", "W-076", "Tan", "Warm"),
            Shade("Warm Chestnut", "W-096", "Deep", "Warm"),
        ],
    ),

    # ---- Too Faced ----
    Product(
        "Too Faced", "Born This Way Natural Finish Foundation",
        finish="Natural", coverage="Medium-Full",
        skin_types="Normal, Combination, Dry", spf=0, fragrance_free=False,
        price_usd=42.0,
        ingredient_notes="Coconut water; alpine rose; hyaluronic acid.",
        shades=[
            Shade("Snow", "SN", "Fair", "Cool"),
            Shade("Almond", "AL", "Light", "Warm"),
            Shade("Warm Beige", "WB", "Light-Medium", "Warm"),
            Shade("Warm Sand", "WS", "Medium", "Warm"),
            Shade("Mocha", "MC", "Tan", "Warm"),
            Shade("Ganache", "GN", "Deep", "Neutral"),
            Shade("Cocoa", "CC", "Rich", "Warm"),
        ],
    ),

    # ---- Tarte ----
    Product(
        "Tarte", "Amazonian Clay Full Coverage Foundation SPF 15",
        finish="Matte", coverage="Full",
        skin_types="Oily, Combination", spf=15, fragrance_free=False,
        price_usd=42.0,
        ingredient_notes="Amazonian clay; mineral SPF; oil-absorbing; 12-hr wear.",
        shades=[
            Shade("Fair", "10N", "Fair", "Neutral"),
            Shade("Light-Medium Neutral", "27N", "Light-Medium", "Neutral"),
            Shade("Medium Sand", "35N", "Medium", "Warm"),
            Shade("Tan Sand", "47S", "Tan", "Warm"),
            Shade("Deep Honey", "56H", "Deep", "Warm"),
            Shade("Rich Golden", "60G", "Rich", "Warm"),
        ],
    ),
    Product(
        "Tarte", "Shape Tape Full Coverage Foundation",
        finish="Matte", coverage="Full",
        skin_types="Combination, Oily", spf=0, fragrance_free=False,
        price_usd=42.0,
        ingredient_notes="Marine plant complex; hyaluronic acid; 16-hr wear.",
        shades=[
            Shade("12N Fair", "12N", "Fair", "Neutral"),
            Shade("22N Light", "22N", "Light", "Neutral"),
            Shade("35H Medium Honey", "35H", "Medium", "Warm"),
            Shade("47S Tan Sand", "47S", "Tan", "Warm"),
            Shade("57G Rich Golden", "57G", "Deep", "Warm"),
        ],
    ),

    # ---- Make Up For Ever ----
    Product(
        "Make Up For Ever", "HD Skin Undetectable Longwear Foundation",
        finish="Natural", coverage="Medium-Full",
        skin_types="All", spf=0, fragrance_free=True,
        price_usd=47.0,
        ingredient_notes="Micro-blurring powder + skincare; 24-hr; undetectable HD finish.",
        shades=[
            Shade("1N00 Alabaster", "1N00", "Fair", "Neutral"),
            Shade("1Y04 Ivory", "1Y04", "Light", "Warm"),
            Shade("2N22 Nude", "2N22", "Light-Medium", "Neutral"),
            Shade("2Y36 Honey", "2Y36", "Medium", "Warm"),
            Shade("3R48 Cinnamon", "3R48", "Tan", "Cool"),
            Shade("3N58 Amber", "3N58", "Deep", "Neutral"),
            Shade("4N74 Chocolate", "4N74", "Rich", "Neutral"),
        ],
    ),
    Product(
        "Make Up For Ever", "Ultra HD Invisible Cover Foundation",
        finish="Radiant", coverage="Medium",
        skin_types="Normal, Combination", spf=0, fragrance_free=False,
        price_usd=47.0,
        ingredient_notes="Hyaluronic spheres; HD pigments for camera-ready finish.",
        shades=[
            Shade("R210 Pink Alabaster", "R210", "Fair", "Cool"),
            Shade("Y245 Soft Sand", "Y245", "Light-Medium", "Warm"),
            Shade("Y415 Almond", "Y415", "Tan", "Warm"),
            Shade("R540 Dark Brown", "R540", "Rich", "Cool"),
        ],
    ),

    # ---- Il Makiage ----
    Product(
        "Il Makiage", "Woke Up Like This Flawless Base Foundation",
        finish="Satin", coverage="Full",
        skin_types="All", spf=0, fragrance_free=False,
        price_usd=44.0,
        ingredient_notes="Hyaluronic acid; caffeine; vitamin E; 24-hr wear.",
        shades=[
            Shade("Shade 20", "20", "Fair", "Neutral"),
            Shade("Shade 55", "55", "Light-Medium", "Warm"),
            Shade("Shade 130", "130", "Medium", "Neutral"),
            Shade("Shade 200", "200", "Tan", "Warm"),
            Shade("Shade 285", "285", "Deep", "Warm"),
            Shade("Shade 325", "325", "Rich", "Neutral"),
        ],
    ),

    # ---- Kosas ----
    Product(
        "Kosas", "Revealer Skin Improving Foundation SPF 25",
        finish="Radiant", coverage="Medium",
        skin_types="All, Dry, Sensitive", spf=25, fragrance_free=True,
        price_usd=42.0,
        ingredient_notes="Hyaluronic acid; niacinamide; peptides; mineral SPF (zinc/titanium).",
        shades=[
            Shade("Tone 1.5", "1.5", "Fair", "Neutral"),
            Shade("Tone 3", "3", "Light", "Warm"),
            Shade("Tone 5", "5", "Light-Medium", "Neutral"),
            Shade("Tone 7", "7", "Medium", "Warm"),
            Shade("Tone 8.5", "8.5", "Medium-Tan", "Warm"),
            Shade("Tone 10", "10", "Deep", "Neutral"),
            Shade("Tone 11", "11", "Rich", "Warm"),
        ],
    ),

    # ---- Ilia ----
    Product(
        "Ilia", "Super Serum Skin Tint SPF 40",
        finish="Dewy", coverage="Sheer",
        skin_types="Dry, Normal, Sensitive", spf=40, fragrance_free=True,
        price_usd=48.0,
        ingredient_notes="Niacinamide; hyaluronic acid; plant-based squalane; non-nano zinc SPF.",
        shades=[
            Shade("Rendezvous ST2", "ST2", "Fair", "Cool"),
            Shade("Bom Bom ST6", "ST6", "Light", "Neutral"),
            Shade("Formentera ST10", "ST10", "Light-Medium", "Warm"),
            Shade("Kingston ST13", "ST13", "Medium", "Warm"),
            Shade("Waikiki ST15", "ST15", "Tan", "Neutral"),
            Shade("Diani ST17", "ST17", "Deep", "Neutral"),
            Shade("Roque ST18", "ST18", "Rich", "Warm"),
        ],
    ),
    Product(
        "Ilia", "True Skin Serum Foundation",
        finish="Natural", coverage="Light-Medium",
        skin_types="Normal, Dry, Sensitive", spf=0, fragrance_free=True,
        price_usd=54.0,
        ingredient_notes="Aloe leaf juice; hyaluronic acid; jojoba; clean-beauty formula.",
        shades=[
            Shade("Wake SF1", "SF1", "Fair", "Neutral"),
            Shade("Reti SF5", "SF5", "Light", "Warm"),
            Shade("Corse SF8", "SF8", "Light-Medium", "Neutral"),
            Shade("Yasawa SF12", "SF12", "Medium-Tan", "Warm"),
            Shade("Iona SF15", "SF15", "Deep", "Neutral"),
        ],
    ),

    # ---- Merit ----
    Product(
        "Merit", "The Minimalist Perfecting Complexion Foundation Stick",
        finish="Natural", coverage="Light-Medium",
        skin_types="All", spf=0, fragrance_free=True,
        price_usd=38.0,
        ingredient_notes="Bakuchiol; niacinamide; buildable stick; clean formula.",
        shades=[
            Shade("Snow", "N05", "Fair", "Neutral"),
            Shade("Buff", "N20", "Light", "Warm"),
            Shade("Beach", "N40", "Light-Medium", "Neutral"),
            Shade("Suede", "N60", "Medium", "Warm"),
            Shade("Cocoa", "N80", "Tan", "Neutral"),
            Shade("Espresso", "N100", "Deep", "Warm"),
        ],
    ),

    # ---- Laura Mercier ----
    Product(
        "Laura Mercier", "Tinted Moisturizer Natural Skin Perfector SPF 30",
        finish="Natural", coverage="Sheer",
        skin_types="Normal, Dry, Sensitive", spf=30, fragrance_free=True,
        price_usd=52.0,
        ingredient_notes="Hydrating tint; chemical SPF; skincare-first formula.",
        shades=[
            Shade("1N1 Porcelain", "1N1", "Fair", "Neutral"),
            Shade("2N1 Nude", "2N1", "Light", "Neutral"),
            Shade("3W1 Bisque", "3W1", "Light-Medium", "Warm"),
            Shade("4W1 Tawny", "4W1", "Medium-Tan", "Warm"),
            Shade("6N1 Mocha", "6N1", "Deep", "Neutral"),
        ],
    ),
    Product(
        "Laura Mercier", "Flawless Lumière Radiance-Perfecting Foundation SPF 30",
        finish="Radiant", coverage="Medium-Full",
        skin_types="Normal, Dry, Combination", spf=30, fragrance_free=True,
        price_usd=52.0,
        ingredient_notes="Hyaluronic acid; vitamin E; luminous SPF 30 finish.",
        shades=[
            Shade("1W1 Ivory", "1W1", "Fair", "Warm"),
            Shade("2C1 Ecru", "2C1", "Light", "Cool"),
            Shade("3N1 Buff", "3N1", "Light-Medium", "Neutral"),
            Shade("4N2 Pecan", "4N2", "Medium-Tan", "Neutral"),
            Shade("5W1 Amber", "5W1", "Tan", "Warm"),
        ],
    ),

    # ---- Hourglass ----
    Product(
        "Hourglass", "Vanish Seamless Finish Foundation Stick",
        finish="Satin", coverage="Full",
        skin_types="Normal, Combination", spf=0, fragrance_free=True,
        price_usd=56.0,
        ingredient_notes="Waterless formula; blurring pigments; second-skin stick.",
        shades=[
            Shade("Vanilla", "VN", "Fair", "Warm"),
            Shade("Shell", "SH", "Light", "Neutral"),
            Shade("Beige", "BG", "Light-Medium", "Neutral"),
            Shade("Sand", "SD", "Medium", "Warm"),
            Shade("Chestnut", "CN", "Tan", "Warm"),
            Shade("Espresso", "EP", "Deep", "Neutral"),
        ],
    ),
    Product(
        "Hourglass", "Ambient Soft Glow Foundation",
        finish="Radiant", coverage="Medium",
        skin_types="Normal, Dry, Combination", spf=0, fragrance_free=True,
        price_usd=59.0,
        ingredient_notes="Photoluminescent technology; soft-focus glow; buildable.",
        shades=[
            Shade("2", "2", "Fair", "Neutral"),
            Shade("4.5", "4.5", "Light-Medium", "Warm"),
            Shade("6", "6", "Medium", "Neutral"),
            Shade("9", "9", "Tan", "Warm"),
            Shade("11", "11", "Deep", "Neutral"),
        ],
    ),

    # ---- Pat McGrath Labs ----
    Product(
        "Pat McGrath Labs", "Skin Fetish: Sublime Perfection Foundation",
        finish="Satin", coverage="Medium-Full",
        skin_types="All", spf=0, fragrance_free=False,
        price_usd=72.0,
        ingredient_notes="Photo-illuminating pigments; hyaluronic acid; skin-cushion complex.",
        shades=[
            Shade("Light 3", "LT3", "Fair", "Cool"),
            Shade("Light 8", "LT8", "Light", "Neutral"),
            Shade("Medium 17", "MD17", "Light-Medium", "Warm"),
            Shade("Medium Deep 24", "MD24", "Medium", "Warm"),
            Shade("Deep 32", "DP32", "Tan", "Neutral"),
            Shade("Deep 36", "DP36", "Deep", "Warm"),
        ],
    ),

    # ---- Huda Beauty ----
    Product(
        "Huda Beauty", "FauxFilter Luminous Matte Foundation",
        finish="Matte", coverage="Full",
        skin_types="Combination, Oily", spf=0, fragrance_free=False,
        price_usd=44.0,
        ingredient_notes="Luminous-matte hybrid pigments; 24-hr wear.",
        shades=[
            Shade("Marshmallow", "100N", "Fair", "Neutral"),
            Shade("Shortbread", "150G", "Light", "Warm"),
            Shade("Sable", "240N", "Light-Medium", "Neutral"),
            Shade("Toasted Coconut", "300G", "Medium", "Warm"),
            Shade("Amaretti", "400N", "Tan", "Neutral"),
            Shade("Toffee", "500G", "Deep", "Warm"),
            Shade("Chocolate Mousse", "540N", "Rich", "Neutral"),
        ],
    ),

    # ---- Milk Makeup ----
    Product(
        "Milk Makeup", "Future Fluid All Over Medium Coverage Hydrating Concealer",
        finish="Natural", coverage="Medium",
        skin_types="Dry, Normal, Combination", spf=0, fragrance_free=True,
        price_usd=26.0,
        ingredient_notes="Hyaluronic acid; concealer-foundation hybrid; vegan.",
        shades=[
            Shade("Fair 1", "F1", "Fair", "Cool"),
            Shade("Light 2", "L2", "Light", "Neutral"),
            Shade("Medium 3", "M3", "Light-Medium", "Warm"),
            Shade("Tan 5", "T5", "Medium-Tan", "Warm"),
            Shade("Deep 7", "D7", "Deep", "Neutral"),
        ],
    ),

    # ---- Glossier ----
    Product(
        "Glossier", "Perfecting Skin Tint",
        finish="Natural", coverage="Sheer",
        skin_types="All", spf=0, fragrance_free=True,
        price_usd=26.0,
        ingredient_notes="Water-based; buildable; barely-there skin tint.",
        shades=[
            Shade("G1", "G1", "Fair", "Neutral"),
            Shade("G4", "G4", "Light-Medium", "Warm"),
            Shade("G7", "G7", "Medium", "Neutral"),
            Shade("G10", "G10", "Tan", "Warm"),
            Shade("G12", "G12", "Deep", "Neutral"),
        ],
    ),
    Product(
        "Glossier", "Stretch Fluid Foundation",
        finish="Radiant", coverage="Light-Medium",
        skin_types="Normal, Dry", spf=0, fragrance_free=True,
        price_usd=34.0,
        ingredient_notes="Snow mushroom; hyaluronic acid; skin-flex polymers.",
        shades=[
            Shade("F1", "F1", "Fair", "Neutral"),
            Shade("L3", "L3", "Light-Medium", "Warm"),
            Shade("M5", "M5", "Medium", "Neutral"),
            Shade("D9", "D9", "Deep", "Warm"),
        ],
    ),

    # ---- e.l.f. ----
    Product(
        "e.l.f.", "Halo Glow Liquid Filter",
        finish="Luminous", coverage="Sheer",
        skin_types="Normal, Combination, Dry", spf=0, fragrance_free=False,
        price_usd=15.0,
        ingredient_notes="Hyaluronic acid; squalane; light-reflecting pearls.",
        shades=[
            Shade("Fair", "1", "Fair", "Neutral"),
            Shade("Light", "2", "Light", "Warm"),
            Shade("Light/Medium", "3", "Light-Medium", "Neutral"),
            Shade("Medium", "4", "Medium", "Warm"),
            Shade("Medium/Tan", "5", "Medium-Tan", "Warm"),
            Shade("Tan", "6", "Tan", "Neutral"),
            Shade("Rich", "7", "Deep", "Warm"),
            Shade("Deep", "8", "Rich", "Neutral"),
        ],
    ),
    Product(
        "e.l.f.", "Camo CC Cream SPF 30",
        finish="Natural", coverage="Full",
        skin_types="All", spf=30, fragrance_free=True,
        price_usd=15.0,
        ingredient_notes="Niacinamide; hyaluronic acid; peptides; chemical SPF 30.",
        shades=[
            Shade("Fair 140N", "140N", "Fair", "Neutral"),
            Shade("Light 240W", "240W", "Light", "Warm"),
            Shade("Medium 340C", "340C", "Light-Medium", "Cool"),
            Shade("Tan 440W", "440W", "Tan", "Warm"),
            Shade("Rich 540W", "540W", "Rich", "Warm"),
        ],
    ),

    # ---- NYX ----
    Product(
        "NYX Professional Makeup", "Can't Stop Won't Stop Full Coverage Foundation",
        finish="Matte", coverage="Full",
        skin_types="Oily, Combination, Normal", spf=0, fragrance_free=False,
        price_usd=15.99,
        ingredient_notes="24-hr wear; waterproof; transfer-resistant film.",
        shades=[
            Shade("Pale", "CSWSF01", "Fair", "Cool"),
            Shade("Light Ivory", "CSWSF04", "Light", "Neutral"),
            Shade("Buff", "CSWSF10", "Light-Medium", "Warm"),
            Shade("Warm Caramel", "CSWSF15", "Medium-Tan", "Warm"),
            Shade("Cappuccino", "CSWSF17.5", "Tan", "Neutral"),
            Shade("Deep Ebony", "CSWSF24", "Rich", "Cool"),
        ],
    ),

    # ---- Revlon ----
    Product(
        "Revlon", "ColorStay Makeup for Combination/Oily Skin",
        finish="Matte", coverage="Medium-Full",
        skin_types="Combination, Oily", spf=15, fragrance_free=False,
        price_usd=14.99,
        ingredient_notes="Oil-controlling; SoftFlex technology; 24-hr wear.",
        shades=[
            Shade("Ivory", "110", "Fair", "Cool"),
            Shade("Buff", "150", "Light", "Neutral"),
            Shade("Natural Beige", "220", "Light-Medium", "Warm"),
            Shade("True Beige", "320", "Medium", "Warm"),
            Shade("Caramel", "400", "Tan", "Warm"),
            Shade("Mahogany", "440", "Deep", "Neutral"),
        ],
    ),

    # ---- CoverGirl ----
    Product(
        "CoverGirl", "Clean Fresh Skin Milk Foundation",
        finish="Dewy", coverage="Sheer",
        skin_types="Normal, Dry, Sensitive", spf=0, fragrance_free=True,
        price_usd=10.99,
        ingredient_notes="Aloe; cucumber; hyaluronic acid; vegan.",
        shades=[
            Shade("Fair 500", "500", "Fair", "Neutral"),
            Shade("Light 510", "510", "Light", "Warm"),
            Shade("Medium 530", "530", "Medium", "Neutral"),
            Shade("Tan 550", "550", "Tan", "Warm"),
            Shade("Deep 570", "570", "Deep", "Warm"),
        ],
    ),

    # ---- Anastasia Beverly Hills ----
    Product(
        "Anastasia Beverly Hills", "Luminous Foundation",
        finish="Luminous", coverage="Medium",
        skin_types="Normal, Dry, Combination", spf=0, fragrance_free=False,
        price_usd=42.0,
        ingredient_notes="Hyaluronic acid; vitamin C; sodium PCA; luminous finish.",
        shades=[
            Shade("110C", "110C", "Fair", "Cool"),
            Shade("220N", "220N", "Light-Medium", "Neutral"),
            Shade("330W", "330W", "Medium", "Warm"),
            Shade("430C", "430C", "Tan", "Cool"),
            Shade("530W", "530W", "Deep", "Warm"),
            Shade("580N", "580N", "Rich", "Neutral"),
        ],
    ),

    # ---- Danessa Myricks ----
    Product(
        "Danessa Myricks Beauty", "Yummy Skin Serum Skin Tint",
        finish="Radiant", coverage="Sheer",
        skin_types="All", spf=0, fragrance_free=True,
        price_usd=42.0,
        ingredient_notes="Hyaluronic acid; niacinamide; vitamin C; skincare-serum tint.",
        shades=[
            Shade("Shade 1", "1", "Fair", "Neutral"),
            Shade("Shade 4", "4", "Light-Medium", "Warm"),
            Shade("Shade 7", "7", "Medium", "Neutral"),
            Shade("Shade 10", "10", "Tan", "Warm"),
            Shade("Shade 14", "14", "Deep", "Neutral"),
        ],
    ),

    # ---- One/Size ----
    Product(
        "One/Size", "Turn Up the Base Versatile Water-Powered Foundation",
        finish="Natural", coverage="Medium-Full",
        skin_types="All", spf=0, fragrance_free=True,
        price_usd=38.0,
        ingredient_notes="Water-based; hyaluronic acid; niacinamide; buildable.",
        shades=[
            Shade("F00", "F00", "Fair", "Neutral"),
            Shade("L10", "L10", "Light", "Warm"),
            Shade("LM15", "LM15", "Light-Medium", "Neutral"),
            Shade("M25", "M25", "Medium", "Warm"),
            Shade("T35", "T35", "Tan", "Neutral"),
            Shade("D45", "D45", "Deep", "Warm"),
            Shade("R55", "R55", "Rich", "Neutral"),
        ],
    ),

    # ---- Westman Atelier ----
    Product(
        "Westman Atelier", "Vital Skincare Complexion Drops",
        finish="Radiant", coverage="Sheer",
        skin_types="Normal, Dry, Combination, Sensitive", spf=0, fragrance_free=True,
        price_usd=88.0,
        ingredient_notes="Bakuchiol; niacinamide; hyaluronic acid; camellia oil; clean.",
        shades=[
            Shade("Atelier I", "I", "Fair", "Neutral"),
            Shade("Atelier IV", "IV", "Light-Medium", "Warm"),
            Shade("Atelier VII", "VII", "Medium-Tan", "Neutral"),
            Shade("Atelier X", "X", "Tan", "Warm"),
            Shade("Atelier XII", "XII", "Deep", "Neutral"),
        ],
    ),

    # ---- Saie ----
    Product(
        "Saie", "Glowy Super Skin Lightweight Serum-Foundation",
        finish="Dewy", coverage="Light-Medium",
        skin_types="Normal, Dry, Sensitive", spf=0, fragrance_free=True,
        price_usd=40.0,
        ingredient_notes="Hyaluronic acid; niacinamide; peptides; clean formula.",
        shades=[
            Shade("100", "100", "Fair", "Neutral"),
            Shade("140", "140", "Light", "Warm"),
            Shade("200", "200", "Light-Medium", "Neutral"),
            Shade("300", "300", "Medium", "Warm"),
            Shade("400", "400", "Tan", "Warm"),
            Shade("500", "500", "Deep", "Neutral"),
        ],
    ),

    # ---- RMS ----
    Product(
        "RMS Beauty", "\"Un\" Cover-Up Cream Foundation",
        finish="Natural", coverage="Medium",
        skin_types="Dry, Normal, Sensitive", spf=0, fragrance_free=True,
        price_usd=48.0,
        ingredient_notes="Organic coconut oil; buriti oil; jojoba; clean/organic.",
        shades=[
            Shade("00", "00", "Fair", "Cool"),
            Shade("22", "22", "Light-Medium", "Warm"),
            Shade("44", "44", "Medium", "Neutral"),
            Shade("77", "77", "Tan", "Warm"),
            Shade("99", "99", "Deep", "Neutral"),
        ],
    ),

    # ---- Jones Road ----
    Product(
        "Jones Road", "What The Foundation",
        finish="Natural", coverage="Sheer",
        skin_types="All", spf=0, fragrance_free=True,
        price_usd=42.0,
        ingredient_notes="Jojoba; grape-seed oil; buildable balm-tint; clean.",
        shades=[
            Shade("00", "00", "Fair", "Neutral"),
            Shade("15", "15", "Light", "Warm"),
            Shade("35", "35", "Light-Medium", "Neutral"),
            Shade("55", "55", "Medium-Tan", "Warm"),
            Shade("75", "75", "Tan", "Warm"),
            Shade("100", "100", "Rich", "Neutral"),
        ],
    ),

    # ---- Physicians Formula ----
    Product(
        "Physicians Formula", "Butter Believe It! Foundation + Concealer",
        finish="Satin", coverage="Medium-Full",
        skin_types="Normal, Dry, Sensitive", spf=0, fragrance_free=True,
        price_usd=15.99,
        ingredient_notes="Murumuru, tucuma, cupuacu butters; hypoallergenic.",
        shades=[
            Shade("Fair to Light", "FL", "Fair", "Cool"),
            Shade("Light to Medium", "LM", "Light", "Warm"),
            Shade("Medium to Tan", "MT", "Medium", "Warm"),
            Shade("Tan to Deep", "TD", "Tan", "Warm"),
        ],
    ),

    # ---- Lancôme ----
    Product(
        "Lancôme", "Teint Idole Ultra Wear Foundation SPF 35",
        finish="Matte", coverage="Full",
        skin_types="Oily, Combination, Normal", spf=35, fragrance_free=False,
        price_usd=55.0,
        ingredient_notes="24-hr wear; SPF 35; hyaluronic acid; camellia extract.",
        shades=[
            Shade("90 Ivoire N", "90N", "Fair", "Neutral"),
            Shade("130 Buff N", "130N", "Light", "Neutral"),
            Shade("310 Bisque C", "310C", "Light-Medium", "Cool"),
            Shade("410 Bisque W", "410W", "Medium", "Warm"),
            Shade("500 Suede W", "500W", "Tan", "Warm"),
            Shade("540 Suede C", "540C", "Deep", "Cool"),
        ],
    ),

    # ---- SHISEIDO ----
    Product(
        "Shiseido", "Synchro Skin Self-Refreshing Foundation SPF 30",
        finish="Natural", coverage="Medium",
        skin_types="Normal, Combination, Oily", spf=30, fragrance_free=True,
        price_usd=52.0,
        ingredient_notes="ActiveForce Technology; sweat- & humidity-proof; SPF 30.",
        shades=[
            Shade("110 Alabaster", "110", "Fair", "Cool"),
            Shade("220 Linen", "220", "Light", "Neutral"),
            Shade("330 Bamboo", "330", "Light-Medium", "Warm"),
            Shade("410 Sunstone", "410", "Medium-Tan", "Warm"),
            Shade("510 Suede", "510", "Deep", "Neutral"),
        ],
    ),
]


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

FIELDNAMES = [
    "brand",
    "product_name",
    "shade_name",
    "shade_code",
    "shade_depth_bucket",
    "undertone_bucket",
    "finish",
    "coverage",
    "skin_type_suitability",
    "spf",
    "fragrance_free",
    "price_usd",
    "ingredient_notes",
]


MAX_SHADES_PER_PRODUCT = 5  # cap to keep total rows within 100-300 target


def _sample_shades(shades: list[Shade], k: int) -> list[Shade]:
    """Evenly sample k shades so depth diversity (fair -> rich) is preserved."""
    if len(shades) <= k:
        return shades
    idxs = [round(i * (len(shades) - 1) / (k - 1)) for i in range(k)]
    seen: set[int] = set()
    picked = []
    for i in idxs:
        if i not in seen:
            seen.add(i)
            picked.append(shades[i])
    return picked


def build_rows() -> list[dict]:
    rows: list[dict] = []
    for p in PRODUCTS:
        k = MAX_SHADES_PER_PRODUCT - 1 if len(p.shades) >= 7 else MAX_SHADES_PER_PRODUCT
        for s in _sample_shades(p.shades, k):
            assert s.depth in DEPTH_BUCKETS, f"Bad depth: {s.depth}"
            assert s.undertone in UNDERTONE_BUCKETS, f"Bad undertone: {s.undertone}"
            assert p.finish in FINISHES, f"Bad finish: {p.finish}"
            assert p.coverage in COVERAGES, f"Bad coverage: {p.coverage}"
            rows.append(
                {
                    "brand": p.brand,
                    "product_name": p.product,
                    "shade_name": s.name,
                    "shade_code": s.code,
                    "shade_depth_bucket": s.depth,
                    "undertone_bucket": s.undertone,
                    "finish": p.finish,
                    "coverage": p.coverage,
                    "skin_type_suitability": p.skin_types,
                    "spf": p.spf,
                    "fragrance_free": p.fragrance_free,
                    "price_usd": p.price_usd,
                    "ingredient_notes": p.ingredient_notes,
                }
            )
    return rows


def main() -> None:
    rows = build_rows()
    out = Path(__file__).parent / "foundations.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    n_products = len(PRODUCTS)
    n_brands = len({p.brand for p in PRODUCTS})
    print(f"Wrote {len(rows)} rows across {n_products} products / {n_brands} brands -> {out}")


if __name__ == "__main__":
    main()
