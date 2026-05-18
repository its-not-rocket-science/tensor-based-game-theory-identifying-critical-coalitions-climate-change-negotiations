"""
ISO 3166-1 alpha-3 resolution for the climate negotiations pipeline.

All four data sources (treaty data, ND-GAIN, WRI Aqueduct, SIDS list) use
different country name conventions. This module resolves any variant to a
canonical ISO3 code so joins work correctly across datasets.

Usage:
    from country_name_resolver import resolve_to_iso3, HISTORICAL_ENTITIES
"""
import pycountry

# Entities present in treaty data with no modern ISO 3166-1 entry.
# Rows for these are excluded before any analysis.
HISTORICAL_ENTITIES: frozenset = frozenset({
    "USSR",
    "Yugoslavia",
    "East Germany",
    "Czechoslovakia",
    "South Vietnam",
    "South Yemen",
    "Republic of South Maluku",
    "South Moluccas, Republic of the ",
    "South Moluccas, Republic of the",
    "Holy See",       # Vatican — not a climate negotiating party in practice
})

# Manual overrides: any name variant -> ISO3 alpha-3.
# Covers names pycountry.countries.lookup() cannot resolve unambiguously.
_MANUAL: dict = {
    # Treaty data colloquial names that differ from ISO standard
    "Bolivia": "BOL",
    "Iran": "IRN",
    "Tanzania": "TZA",
    "Venezuela": "VEN",
    "Syria": "SYR",
    "Laos": "LAO",
    "Russia": "RUS",
    "South Korea": "KOR",
    "North Korea": "PRK",
    "Cape Verde": "CPV",
    "Cabo Verde": "CPV",
    "Micronesia": "FSM",
    "Swaziland": "SWZ",          # now Eswatini
    "Macedonia": "MKD",           # now North Macedonia
    "Czech Republic": "CZE",
    "Czechia": "CZE",
    "Cote d'Ivoire": "CIV",
    "Ivory Coast": "CIV",
    "Congo, Democratic Republic of": "COD",
    "Congo, Republic of the": "COG",
    "Republic of the Congo": "COG",
    "Korea, Democratic People's Republic": "PRK",
    "Korea, Republic of": "KOR",
    "Viet Nam": "VNM",
    "Vietnam": "VNM",
    "Kosovo": "XKX",              # no official ISO3; use UNSD provisional
    "Turkey": "TUR",              # officially renamed to Türkiye in 2022
    "Turkiye": "TUR",
    "Türkiye": "TUR",
    "Cote d'ivoire": "CIV",       # treaty data uses lowercase 'i'
    "Côte D'Ivoire": "CIV",
    # ND-GAIN official long-form names
    "Bolivia, Plurinational State of": "BOL",
    "Iran, Islamic Republic of": "IRN",
    "Tanzania, United Republic of": "TZA",
    "Venezuela, Bolivarian Republic of": "VEN",
    "Syrian Arab Republic": "SYR",
    "Lao People's Democratic Republic": "LAO",
    "Russian Federation": "RUS",
    "Korea, Democratic People's Republic of": "PRK",
    "Micronesia, Federated States of": "FSM",
    "Eswatini": "SWZ",
    "North Macedonia": "MKD",
    "Côte d'Ivoire": "CIV",
    "Congo, Democratic Republic of the": "COD",
    "Congo, The Democratic Republic of the": "COD",
    "Palestine, State of": "PSE",
    "Palestine": "PSE",
    # WRI Aqueduct variant names
    "Brunei": "BRN",
    "Brunei Darussalam": "BRN",
    "Dem. Rep. Congo": "COD",
    "Congo, Dem. Rep.": "COD",
    "Democratic Republic of the Congo": "COD",
    "Rep. Congo": "COG",
    "Republic of Congo": "COG",
    "Korea, Dem. People's Rep.": "PRK",
    "Korea, Rep.": "KOR",
    "Slovak Republic": "SVK",
    "Kyrgyz Republic": "KGZ",
    "Sao Tome and Principe": "STP",
    "São Tomé and Príncipe": "STP",
}


# G20 members by ISO3 — use for filtering instead of name strings.
# Covers "Russian Federation" vs "Russia", "Korea, Republic of" vs "South Korea", etc.
G20_ISO3: frozenset = frozenset({
    "ARG", "AUS", "BRA", "CAN", "CHN", "FRA", "DEU",
    "IND", "IDN", "ITA", "JPN", "MEX", "RUS", "SAU",
    "ZAF", "KOR", "TUR", "GBR", "USA",
    # EU has no single ISO3; member states handled individually above
})


def resolve_to_iso3(name: str):
    """
    Resolve a country name to ISO3 alpha-3 code.

    Returns None for historical entities or names that cannot be resolved.
    """
    if not isinstance(name, str):
        return None
    name = name.strip().replace('\xa0', ' ')  # normalize non-breaking spaces from Excel
    if name in HISTORICAL_ENTITIES:
        return None
    if name in _MANUAL:
        return _MANUAL[name]
    try:
        return pycountry.countries.lookup(name).alpha_3
    except LookupError:
        return None


def build_iso3_to_name(country_names: list) -> dict:
    """Build ISO3 -> display name from a list of names (last write wins on collision)."""
    result = {}
    for name in country_names:
        iso3 = resolve_to_iso3(name)
        if iso3 is not None:
            result[iso3] = name
    return result
