PARAMETERS = {'target_height': 100.0, 'wall_thickness': 3.0, 'bowl_start_z': 14.0, 'body_base_radius': 41.0, 'body_lower_z': 25.0, 'body_lower_radius': 43.0, 'body_mid_z': 55.0, 'body_mid_radius': 46.5, 'body_upper_z': 80.0, 'body_upper_radius': 50.0, 'top_outer_radius': 53.0, 'foot_height': 16.0, 'foot_bottom_contact_radius': 45.5, 'foot_max_radius': 47.0, 'foot_bulge_z': 3.0, 'foot_shoulder_z': 11.0, 'foot_shoulder_radius': 44.0, 'foot_top_radius': 41.5, 'handle_center_x': 61.0, 'handle_center_z': 58.0, 'handle_outer_rx': 40.0, 'handle_outer_rz': 36.0, 'handle_band_width': 12.0, 'handle_depth': 14.0, 'handle_attachment_x': 40.0, 'handle_clip_length': 80.0, 'cut_clearance': 2.0}
import cadquery as cq
target_height = PARAMETERS['target_height']
wall_thickness = PARAMETERS['wall_thickness']
bowl_start_z = PARAMETERS['bowl_start_z']
body_base_radius = PARAMETERS['body_base_radius']
body_lower_z = PARAMETERS['body_lower_z']
body_lower_radius = PARAMETERS['body_lower_radius']
body_mid_z = PARAMETERS['body_mid_z']
body_mid_radius = PARAMETERS['body_mid_radius']
body_upper_z = PARAMETERS['body_upper_z']
body_upper_radius = PARAMETERS['body_upper_radius']
top_outer_radius = PARAMETERS['top_outer_radius']
foot_height = PARAMETERS['foot_height']
foot_bottom_contact_radius = PARAMETERS['foot_bottom_contact_radius']
foot_max_radius = PARAMETERS['foot_max_radius']
foot_bulge_z = PARAMETERS['foot_bulge_z']
foot_shoulder_z = PARAMETERS['foot_shoulder_z']
foot_shoulder_radius = PARAMETERS['foot_shoulder_radius']
foot_top_radius = PARAMETERS['foot_top_radius']
handle_center_x = PARAMETERS['handle_center_x']
handle_center_z = PARAMETERS['handle_center_z']
handle_outer_rx = PARAMETERS['handle_outer_rx']
handle_outer_rz = PARAMETERS['handle_outer_rz']
handle_band_width = PARAMETERS['handle_band_width']
handle_depth = PARAMETERS['handle_depth']
handle_attachment_x = PARAMETERS['handle_attachment_x']
handle_clip_length = PARAMETERS['handle_clip_length']
cut_clearance = PARAMETERS['cut_clearance']
rim_base_z = target_height - wall_thickness / 2.0
rim_mid_radius = top_outer_radius - wall_thickness / 2.0
inner_bottom_z = bowl_start_z + wall_thickness
inner_base_radius = body_base_radius - wall_thickness
inner_lower_radius = body_lower_radius - wall_thickness
inner_mid_radius = body_mid_radius - wall_thickness
inner_upper_radius = body_upper_radius - wall_thickness
inner_top_radius = top_outer_radius - wall_thickness
handle_inner_rx = handle_outer_rx - handle_band_width
handle_inner_rz = handle_outer_rz - handle_band_width
cup_shell = cq.Workplane('XZ').moveTo(0.0, bowl_start_z).lineTo(body_base_radius, bowl_start_z).spline([(body_lower_radius, body_lower_z), (body_mid_radius, body_mid_z), (body_upper_radius, body_upper_z), (top_outer_radius, rim_base_z)], includeCurrent=True).threePointArc((rim_mid_radius, target_height), (inner_top_radius, rim_base_z)).spline([(inner_upper_radius, body_upper_z), (inner_mid_radius, body_mid_z), (inner_lower_radius, body_lower_z), (inner_base_radius, inner_bottom_z)], includeCurrent=True).lineTo(0.0, inner_bottom_z).close().revolve(360.0, (0.0, 0.0), (0.0, 1.0))
foot = cq.Workplane('XZ').moveTo(0.0, 0.0).lineTo(foot_bottom_contact_radius, 0.0).spline([(foot_max_radius, foot_bulge_z), (foot_shoulder_radius, foot_shoulder_z), (foot_top_radius, foot_height)], includeCurrent=True).lineTo(0.0, foot_height).close().revolve(360.0, (0.0, 0.0), (0.0, 1.0))
handle_outer = cq.Workplane('XZ').center(handle_center_x, handle_center_z).ellipse(handle_outer_rx, handle_outer_rz).extrude(handle_depth / 2.0, both=True)
handle_inner = cq.Workplane('XZ').center(handle_center_x, handle_center_z).ellipse(handle_inner_rx, handle_inner_rz).extrude((handle_depth + cut_clearance) / 2.0, both=True)
handle_ring = handle_outer.cut(handle_inner)
handle_clip = cq.Workplane('XY').box(handle_clip_length, handle_depth + cut_clearance, target_height + 2.0 * cut_clearance).translate((handle_attachment_x + handle_clip_length / 2.0, 0.0, target_height / 2.0))
handle = handle_ring.intersect(handle_clip)
r = cup_shell.union(foot).union(handle).clean()
