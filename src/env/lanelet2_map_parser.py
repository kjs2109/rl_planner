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
    
    def get_zoomed_area(self, center, zoom_width, zoom_height):
        # map coordinates 
        cx, cy, heading = center
        zoom_area = box(cx - zoom_width / 2, cy - zoom_height / 2,
                        cx + zoom_width / 2, cy + zoom_height / 2).buffer(5.0)

        # transform lanelet map -> ego_center 
        lanelet_polys = [
            translate(rotate(poly, math.pi/2-heading, origin=(cx, cy), use_radians=True), xoff=-cx, yoff=-cy) 
            for poly in self.lanelet_polygons
        ]
        zoom_area = translate(zoom_area, xoff=-cx, yoff=-cy) 

        selected_polys = [poly for poly in lanelet_polys if poly.intersects(zoom_area)]

        if lanelet_polys:
            drivable_area = unary_union(selected_polys)
        else:
            drivable_area = None

        if drivable_area and not drivable_area.is_empty:
            non_drivable_area = zoom_area.difference(drivable_area)
        else:
            non_drivable_area = zoom_area

        return drivable_area, non_drivable_area
    
    def get_parking_area(self, center, zoom_width=60, zoom_height=90): 
        cx, cy, heading = center 
        zoom_area = box(cx - zoom_width / 2, cy - zoom_height / 2,
                        cx + zoom_width / 2, cy + zoom_height / 2).buffer(5.0)
        return [
            translate(rotate(poly, math.pi/2-heading, origin=(cx, cy), use_radians=True), xoff=-cx, yoff=-cy)
            for poly in self.parking_areas if poly.intersects(zoom_area)
        ]
    
    def get_parking_drivable_area(self, center, zoom_width, zoom_height):
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
        
    def _get_boundary_coordinates(self, area):
        coords = []
        if area is not None:
            if isinstance(area, Polygon):
                coords.append(area.exterior.coords)

            elif isinstance(area, MultiPolygon):
                for poly in area.geoms:
                    coords.append(poly.exterior.coords)
        return coords 

    def visualize(self):
        polygons = self.lanelet_polygons
        fig, ax = plt.subplots()
        for poly in polygons:
            x, y = poly.exterior.xy
            ax.plot(x, y, color="gray")
        ax.set_aspect("equal")
        ax.set_title("Lanelet2 Map Visualization")
        plt.grid(True)
        plt.show()

    def visualize_zoomed(self, center: Tuple[float, float], zoom_width: float = 50.0, zoom_height: float = 50.0):
        cx, cy = center
        zoom_area = box(cx - zoom_width / 2, cy - zoom_height / 2,
                        cx + zoom_width / 2, cy + zoom_height / 2)

        zoomed_polygons = [poly for poly in self.lanelet_polygons if poly.intersects(zoom_area)]

        fig, ax = plt.subplots()
        for poly in zoomed_polygons:
            patch = plt.Polygon(list(poly.exterior.coords), closed=True, facecolor="lightgray", edgecolor="gray", linewidth=1)
            ax.add_patch(patch)

        ax.set_xlim(cx - zoom_width / 2, cx + zoom_width / 2)
        ax.set_ylim(cy - zoom_height / 2, cy + zoom_height / 2)
        ax.set_aspect("equal")
        ax.set_title(f"Zoomed View around ({cx:.2f}, {cy:.2f})")
        plt.grid(True)
        plt.show()

    def visualize_drivable_area(self, center: Tuple[float, float, float], zoom_width: float = 50.0, zoom_height: float = 50.0):
        cx, cy, heading = center

        zoom_area = box(cx - zoom_width / 2, cy - zoom_height / 2,
                        cx + zoom_width / 2, cy + zoom_height / 2)

        safety_margin = 10.0 
        expanded_zoom_area = zoom_area.buffer(safety_margin)
        translated_zoom_area = translate(zoom_area, xoff=-cx, yoff=-cy)

        # zoom 영역과 교차하는 polygon만 rotate
        rotated_lanelet_polys = [
            translate(rotate(poly, heading, origin=(cx, cy), use_radians=True), xoff=-cx, yoff=-cy)
            for poly in self.lanelet_polygons
            if poly.intersects(expanded_zoom_area)
        ]

        if rotated_lanelet_polys:
            drivable_area = unary_union(rotated_lanelet_polys)
        else:
            drivable_area = None

        if drivable_area and not drivable_area.is_empty:
            non_drivable_area = translated_zoom_area.difference(drivable_area)
        else:
            non_drivable_area = translated_zoom_area

        fig, ax = plt.subplots()
        if isinstance(non_drivable_area, Polygon):
            patch = plt.Polygon(list(non_drivable_area.exterior.coords), closed=True, facecolor="lightgray", edgecolor="gray", alpha=1.0)
            ax.add_patch(patch)
        else:  
            for nd in non_drivable_area.geoms:
                patch = plt.Polygon(list(nd.exterior.coords), closed=True, facecolor="lightgray", edgecolor="gray", alpha=1.0)
                ax.add_patch(patch)
        
        drivable_area_coords = self._get_boundary_coordinates(drivable_area) 
        for coords in drivable_area_coords:
            xs, ys = zip(*coords)
            ax.scatter(xs, ys, color='red', s=10) 

        # 주행 가능 영역 시각화 
        # if drivable_area and not drivable_area.is_empty:
        #     if isinstance(drivable_area, Polygon):
        #         patch = plt.Polygon(list(drivable_area.exterior.coords), closed=True, facecolor="lightgreen", edgecolor="green", alpha=0.7)
        #         ax.add_patch(patch)
        #     else:  
        #         for d in drivable_area.geoms:
        #             patch = plt.Polygon(list(d.exterior.coords), closed=True, facecolor="lightgreen", edgecolor="green", alpha=0.7)
        #             ax.add_patch(patch)

        ax.set_xlim(-zoom_width//2, zoom_width//2) 
        ax.set_ylim(-zoom_height//2, zoom_height//2)
        ax.set_xticks(np.arange(-zoom_width//2, zoom_width//2+1, 10))  
        ax.set_yticks(np.arange(-zoom_height//2, zoom_height//2+1, 10))  
        ax.set_aspect('equal')
        ax.set_title(f"Drivable Area ({cx}, {cy}, {np.degrees(heading):.1f}°)")
        plt.grid(True)
        plt.show()

    def visualize_areas(self):
        fig, ax = plt.subplots()

        print(f"Lanelets: {len(self.lanelet_polygons)}, Parking_lot: {len(self.parking_lots)}, Parking: {len(self.parking_areas)}, Obstacle: {len(self.obstacle_areas)}")

        # 1. Lanelet polygons (회색)
        for poly in self.lanelet_polygons:
            patch = plt.Polygon(list(poly.exterior.coords), closed=True,
                                facecolor="lightgray", edgecolor="gray", alpha=0.7, label="Lanelet")
            ax.add_patch(patch)
        
        # 2. Parking lot areas (연두색)
        for poly in self.parking_lots:
            patch = plt.Polygon(list(poly.exterior.coords), closed=True,
                                facecolor="lightgreen", edgecolor="green", alpha=0.5, label="Parking Lot")
            ax.add_patch(patch)


        # 3. Parking areas (연한 파란색)
        for poly in self.parking_areas:
            patch = plt.Polygon(list(poly.exterior.coords), closed=True,
                                facecolor="lightblue", edgecolor="blue", alpha=0.5, label="Parking Area")
            ax.add_patch(patch)

        # 4. Obstacle areas (연한 빨간색)
        for poly in self.obstacle_areas:
            patch = plt.Polygon(list(poly.exterior.coords), closed=True,
                                facecolor="lightcoral", edgecolor="red", alpha=0.5, label="Obstacle Area")
            ax.add_patch(patch)

        #  all_geoms = self.lanelet_polygons + self.parking_areas + self.obstacle_areas
        all_geoms = self.parking_areas 
        if all_geoms:
            bounds = unary_union(all_geoms).bounds
            minx, miny, maxx, maxy = bounds
            ax.set_xlim(minx - 10, maxx + 10)
            ax.set_ylim(miny - 10, maxy + 10)

        ax.set_aspect("equal")
        ax.set_title("Lanelet, Parking, and Obstacle Areas")
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    parser = LaneletMapParser("../../data/lanelet2_map/parking_lot_lanelet2_map_v1.osm")
    parser.visualize()
    parser.visualize_areas()
    parser.visualize_zoomed(center=(99230, 34950), zoom_width=50, zoom_height=50)
    parser.visualize_drivable_area(center=(99240.5859375, 34947.734375, -0.35305924412371), zoom_width=40, zoom_height=60) 