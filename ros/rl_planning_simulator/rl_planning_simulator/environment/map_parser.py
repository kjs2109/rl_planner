import math 
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List

import xml.etree.ElementTree as ET
from shapely.geometry import box
from shapely.ops import unary_union
from shapely.affinity import rotate, translate 
from shapely.geometry import Polygon, MultiPolygon


class LaneletMapParser:
    def __init__(self, osm_path: str):
        self.osm_path = osm_path
        self.node_dict: Dict[int, Tuple[float, float]] = {}
        self.ways: Dict[int, List[int]] = {}
        self.parking_areas: List[Polygon] = [] 
        self.obstacle_areas: List[Polygon] = []
        self.parking_lots: List[Polygon] = []
        self.lanelets: List[Dict[str, List[int]]] = []

        self._parse_osm()
        self.lanelet_polygons = self._get_lanelet_polygons()

    def _parse_osm(self):
        tree = ET.parse(self.osm_path)
        root = tree.getroot()

        # node_id 별로 local 좌표 저장 
        for node in root.findall("node"):
            node_id = int(node.attrib["id"])
            tags = {tag.attrib["k"]: float(tag.attrib["v"]) for tag in node.findall("tag")}
            if "local_x" in tags and "local_y" in tags:
                self.node_dict[node_id] = (tags["local_x"], tags["local_y"])

        # way_id 별로 참조 node_id 저장
        # way 기반 detection_area
        for way in root.findall("way"):
            way_id = int(way.attrib["id"])
            tags = {tag.attrib["k"]: tag.attrib["v"] for tag in way.findall("tag")}
            nds = [int(nd.attrib["ref"]) for nd in way.findall("nd")]
            self.ways[way_id] = nds

            if tags.get("type") == "detection_area":
                coords = [self.node_dict[nid] for nid in nds if nid in self.node_dict]
                polygon = Polygon(coords).buffer(0)
                if tags.get("subtype") == "parking_area":
                    self.parking_areas.append(polygon)
                elif tags.get("subtype") == "parking_lot":
                    self.parking_lots.append(polygon)

        # relation 기반 multipolygon
        for rel in root.findall("relation"):
            tags = {tag.attrib["k"]: tag.attrib["v"] for tag in rel.findall("tag")}
            if tags.get("type") == "multipolygon":
                role_outers = [int(m.attrib["ref"]) for m in rel.findall("member") if m.attrib["role"] == "outer"]
                coords = self._get_coords_from_way_ids(role_outers)
                polygon = Polygon(coords).buffer(0)
                if tags.get("subtype") == "obstacle_area":
                    self.obstacle_areas.append(polygon)


        # 모든 relation 태그를 돌며 차선의 양쪽(left, right) 참조 way_id 저장 
        for rel in root.findall("relation"):
            tags = {tag.attrib["k"]: tag.attrib["v"] for tag in rel.findall("tag")}
            if tags.get("type") == "lanelet":
                lanelet = {"left": [], "right": []}
                for member in rel.findall("member"):
                    role = member.attrib["role"]
                    ref = int(member.attrib["ref"])
                    if role in lanelet:
                        lanelet[role].append(ref)
                self.lanelets.append(lanelet)

    def _get_lanelet_polygons(self) -> List[Polygon]:
        polygons = []
        for lanelet in self.lanelets:
            left_coords = self._get_coords_from_way_ids(lanelet["left"])
            right_coords = self._get_coords_from_way_ids(lanelet["right"])
            if left_coords and right_coords:
                full_coords = left_coords + right_coords[::-1]
                polygon = Polygon(full_coords).buffer(0.1)  
                polygons.append(polygon)
        return polygons

    def _get_coords_from_way_ids(self, way_ids:List[int]) -> List[Tuple[float, float]]:
        coords = []
        for wid in way_ids:
            node_ids = self.ways.get(wid, [])
            coords += [self.node_dict[nid] for nid in node_ids if nid in self.node_dict]
        return coords
    
    def get_non_drivable_area(self, margin:float=20.0):

        if not self.lanelet_polygons:
            raise ValueError("No lanelet polygons parsed.")

        drivable_area = unary_union(self.lanelet_polygons)

        minx, miny, maxx, maxy = drivable_area.bounds
        map_bounds = box(minx - margin, miny - margin, maxx + margin, maxy + margin)

        non_drivable_area = map_bounds.difference(drivable_area)

        return non_drivable_area
    
    def get_semantic_areas(self, center, zoom_width, zoom_height):
        cx, cy, heading = center
        zoom_area = box(cx - zoom_width / 2, cy - zoom_height / 2,
                        cx + zoom_width / 2, cy + zoom_height / 2).buffer(5.0)

        def rotate_and_translate(polys):
            return [
                translate(rotate(poly, math.pi/2-heading, origin=(cx, cy), use_radians=True), xoff=-cx, yoff=-cy)
                for poly in polys if poly.intersects(zoom_area) 
            ]

        lanelet_area = unary_union(rotate_and_translate(self.lanelet_polygons))
        parking_lot_area = unary_union(rotate_and_translate(self.parking_lots))
        obstacle_area = unary_union(rotate_and_translate(self.obstacle_areas))

        drivable_area = unary_union([lanelet_area, parking_lot_area])
        non_drivable_area = obstacle_area

        if not drivable_area or drivable_area.is_empty:
            drivable_area = None
        if not non_drivable_area or non_drivable_area.is_empty:
            non_drivable_area = None

        translated_zoom_area = translate(zoom_area, xoff=-cx, yoff=-cy)

        zoomed_non_drivable_area = translated_zoom_area.difference(drivable_area) if drivable_area else translated_zoom_area
        if non_drivable_area:
            zoomed_non_drivable_area = unary_union([zoomed_non_drivable_area, non_drivable_area])

        return drivable_area, zoomed_non_drivable_area, lanelet_area