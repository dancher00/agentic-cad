PARAMETERS = {'target_height': 120.0, 'wall_thickness': 0.2, 'body_diameter': 74.9, 'neck_diameter': 65.8, 'shoulder_start_height': 99.0, 'neck_start_height': 116.0, 'base_chime_height': 6.0, 'base_contact_diameter': 63.0, 'base_recess_depth': 2.8, 'base_recess_flat_diameter': 40.0, 'lid_top_height': 118.6, 'rim_outer_diameter': 68.0, 'rim_roll_diameter': 2.2, 'rim_flange_outer_diameter': 66.2, 'rim_flange_bottom_height': 118.35, 'rim_flange_thickness': 0.45}
import cadquery as cq
target_height = PARAMETERS['target_height']
wall_thickness = PARAMETERS['wall_thickness']
body_diameter = PARAMETERS['body_diameter']
neck_diameter = PARAMETERS['neck_diameter']
shoulder_start_height = PARAMETERS['shoulder_start_height']
neck_start_height = PARAMETERS['neck_start_height']
base_chime_height = PARAMETERS['base_chime_height']
base_contact_diameter = PARAMETERS['base_contact_diameter']
base_recess_depth = PARAMETERS['base_recess_depth']
base_recess_flat_diameter = PARAMETERS['base_recess_flat_diameter']
lid_top_height = PARAMETERS['lid_top_height']
rim_outer_diameter = PARAMETERS['rim_outer_diameter']
rim_roll_diameter = PARAMETERS['rim_roll_diameter']
rim_flange_outer_diameter = PARAMETERS['rim_flange_outer_diameter']
rim_flange_bottom_height = PARAMETERS['rim_flange_bottom_height']
rim_flange_thickness = PARAMETERS['rim_flange_thickness']
NON_PENETRATION_CAVITY = cq.Workplane('XZ').moveTo(0.0, base_recess_depth + wall_thickness).lineTo(base_recess_flat_diameter / 2.0, base_recess_depth + wall_thickness).spline([(base_recess_flat_diameter / 2.0, base_recess_depth + wall_thickness), (body_diameter / 2.0 - 8.0, base_recess_depth + wall_thickness + 0.4), (body_diameter / 2.0 - 2.0, base_chime_height - 1.0), (body_diameter / 2.0 - wall_thickness, base_chime_height + wall_thickness)], tangents=((1.0, 0.0), (0.0, 1.0))).lineTo(body_diameter / 2.0 - wall_thickness, shoulder_start_height).spline([(body_diameter / 2.0 - wall_thickness, shoulder_start_height), (body_diameter / 2.0 - wall_thickness - 0.1, shoulder_start_height + 2.0), (body_diameter / 2.0 - wall_thickness - 0.8, shoulder_start_height + 5.0), (neck_diameter / 2.0 + 1.0, neck_start_height - 3.0), (neck_diameter / 2.0 - wall_thickness, neck_start_height)], tangents=((0.0, 1.0), (0.0, 1.0))).lineTo(neck_diameter / 2.0 - wall_thickness, lid_top_height - wall_thickness).lineTo(0.0, lid_top_height - wall_thickness).close().revolve(360.0, (0.0, 0.0), (0.0, 1.0))
outer_blank = cq.Workplane('XZ').moveTo(0.0, 0.0).lineTo(base_contact_diameter / 2.0, 0.0).spline([(base_contact_diameter / 2.0, 0.0), (body_diameter / 2.0 - 1.0, 1.4), (body_diameter / 2.0, base_chime_height)], tangents=((1.0, 0.0), (0.0, 1.0))).lineTo(body_diameter / 2.0, shoulder_start_height).spline([(body_diameter / 2.0, shoulder_start_height), (body_diameter / 2.0 - 0.1, shoulder_start_height + 2.0), (body_diameter / 2.0 - 0.8, shoulder_start_height + 5.0), (neck_diameter / 2.0 + 1.2, neck_start_height - 3.0), (neck_diameter / 2.0, neck_start_height)], tangents=((0.0, 1.0), (0.0, 1.0))).lineTo(neck_diameter / 2.0, lid_top_height).lineTo(0.0, lid_top_height).close().revolve(360.0, (0.0, 0.0), (0.0, 1.0))
base_recess_cut = cq.Workplane('XZ').moveTo(0.0, -1.0).lineTo(base_contact_diameter / 2.0, -1.0).threePointArc((base_contact_diameter / 2.0 - 4.9, 1.6), (base_recess_flat_diameter / 2.0, base_recess_depth)).lineTo(0.0, base_recess_depth).close().revolve(360.0, (0.0, 0.0), (0.0, 1.0))
rim_flange = cq.Workplane('XY').workplane(offset=rim_flange_bottom_height).circle(rim_flange_outer_diameter / 2.0).circle(neck_diameter / 2.0 - wall_thickness * 2.0).extrude(rim_flange_thickness)
rim_roll_outer = cq.Workplane('XZ').moveTo(rim_outer_diameter / 2.0 - rim_roll_diameter / 2.0, target_height - rim_roll_diameter / 2.0).circle(rim_roll_diameter / 2.0).revolve(360.0, (0.0, 0.0), (0.0, 1.0))
rim_roll_core = cq.Workplane('XZ').moveTo(rim_outer_diameter / 2.0 - rim_roll_diameter / 2.0, target_height - rim_roll_diameter / 2.0).circle(rim_roll_diameter / 2.0 - wall_thickness).revolve(360.0, (0.0, 0.0), (0.0, 1.0))
external = outer_blank.cut(base_recess_cut).union(rim_flange).union(rim_roll_outer)
r = external.cut(rim_roll_core).cut(NON_PENETRATION_CAVITY)
