# -*- coding: utf-8 -*-

def get_data_config(data_types, source):
    if data_types[0] == "position" and len(data_types) == 1:
        as_3D = True
    else:
        as_3D = False
    return {
        "data_source": source,
        "data_types": data_types,
        "as_3D": as_3D,
        "unit_bounds": True
    }
