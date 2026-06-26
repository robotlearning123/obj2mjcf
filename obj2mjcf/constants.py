# Indentation level for the generated XML.
XML_INDENTATION = "  "

# Character used to denote a comment in an MTL file.
MTL_COMMENT_CHAR = "#"

# Maps an MTL keyword to the ``Material`` attribute it populates. Several keywords
# (the normal/bump aliases) map to a single attribute. Matching is done on the exact
# whitespace-delimited keyword, not a prefix, to avoid collisions (e.g. ``disp`` vs ``d``).
MTL_KEYWORD_TO_ATTR = {
    # Phong / classic fields (MuJoCo).
    "Ka": "Ka",  # ambient color
    "Kd": "Kd",  # diffuse color
    "Ks": "Ks",  # specular color
    "Ke": "Ke",  # emissive color
    "d": "d",  # dissolve (alpha)
    "Tr": "Tr",  # transparency (1 - alpha)
    "Ns": "Ns",  # specular exponent (shininess)
    # PBR extension scalars.
    "Pr": "Pr",  # roughness
    "Pm": "Pm",  # metallic
    # Texture maps.
    "map_Kd": "map_Kd",  # diffuse / albedo
    "map_Ks": "map_Ks",  # specular
    "map_Ke": "map_Ke",  # emissive
    "map_Pr": "map_Pr",  # roughness
    "map_Pm": "map_Pm",  # metallic
    "map_d": "map_d",  # opacity
    "norm": "map_norm",  # normal (PBR extension)
    "map_Bump": "map_norm",  # normal (bump alias)
    "map_bump": "map_norm",
    "bump": "map_norm",
}

# Backwards-compatible tuple of the classic MuJoCo-relevant fields.
MTL_FIELDS = ("Ka", "Kd", "Ks", "d", "Tr", "Ns", "map_Kd")

# Default UsdPreviewSurface values used when a material does not specify them.
USD_DEFAULT_DIFFUSE = (0.8, 0.8, 0.8)
USD_DEFAULT_ROUGHNESS = 0.5
USD_DEFAULT_METALLIC = 0.0
