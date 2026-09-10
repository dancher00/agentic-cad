PARAMETERS = {'target_height': 100.0, 'wall_thickness': 3.0, 'bowl_start_z': 14.0, 'body_base_radius': 41.0, 'body_lower_z': 25.0, 'body_lower_radius': 43.0, 'body_mid_z': 55.0, 'body_mid_radius': 46.5, 'body_upper_z': 80.0, 'body_upper_radius': 50.0, 'top_outer_radius': 53.0, 'foot_height': 16.0, 'foot_bottom_contact_radius': 45.5, 'foot_max_radius': 47.0, 'foot_bulge_z': 3.0, 'foot_shoulder_z': 11.0, 'foot_shoulder_radius': 44.0, 'foot_top_radius': 41.5, 'handle_depth': 10.0, 'handle_edge_radius': 2.0}
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
rim_base_z = target_height - wall_thickness / 2.0
rim_mid_radius = top_outer_radius - wall_thickness / 2.0
inner_bottom_z = bowl_start_z + wall_thickness
inner_base_radius = body_base_radius - wall_thickness
inner_lower_radius = body_lower_radius - wall_thickness
inner_mid_radius = body_mid_radius - wall_thickness
inner_upper_radius = body_upper_radius - wall_thickness
inner_top_radius = top_outer_radius - wall_thickness
cup_shell = cq.Workplane('XZ').moveTo(0.0, bowl_start_z).lineTo(body_base_radius, bowl_start_z).spline([(body_lower_radius, body_lower_z), (body_mid_radius, body_mid_z), (body_upper_radius, body_upper_z), (top_outer_radius, rim_base_z)], includeCurrent=True).threePointArc((rim_mid_radius, target_height), (inner_top_radius, rim_base_z)).spline([(inner_upper_radius, body_upper_z), (inner_mid_radius, body_mid_z), (inner_lower_radius, body_lower_z), (inner_base_radius, inner_bottom_z)], includeCurrent=True).lineTo(0.0, inner_bottom_z).close().revolve(360.0, (0.0, 0.0), (0.0, 1.0))
foot = cq.Workplane('XZ').moveTo(0.0, 0.0).lineTo(foot_bottom_contact_radius, 0.0).spline([(foot_max_radius, foot_bulge_z), (foot_shoulder_radius, foot_shoulder_z), (foot_top_radius, foot_height)], includeCurrent=True).lineTo(0.0, foot_height).close().revolve(360.0, (0.0, 0.0), (0.0, 1.0))
# Photo-reviewed contour in the handle plane, at the 100 mm reference height.
handle_depth = PARAMETERS['handle_depth']
handle_edge_radius = PARAMETERS['handle_edge_radius']
handle_outer = (cq.Workplane('XZ').moveTo(-46, 86)
    .lineTo(-79, 86).threePointArc((-84, 84), (-83, 79))
    .spline([(-78, 63), (-70, 49), (-61, 43), (-40, 33)], includeCurrent=True)
    .lineTo(-35, 48).lineTo(-43, 70).close().extrude(handle_depth / 2, both=True))
NON_PENETRATION_HANDLE_APERTURE = (cq.Workplane('XZ').moveTo(-44, 66)
    .spline([(-48, 66), (-57, 74), (-63, 75), (-67, 72), (-67, 65),
             (-63, 57), (-57, 53), (-51, 53), (-45, 58)], includeCurrent=True)
    .close().extrude(handle_depth, both=True).translate((-4, 0, 0)))
handle_outer = handle_outer.edges('not |Y').fillet(handle_edge_radius)
handle = handle_outer.cut(NON_PENETRATION_HANDLE_APERTURE)
NON_PENETRATION_CAVITY = (cq.Workplane('XZ').moveTo(0, inner_bottom_z)
    .lineTo(inner_base_radius, inner_bottom_z)
    .spline([(inner_lower_radius, body_lower_z), (inner_mid_radius, body_mid_z),
             (inner_upper_radius, body_upper_z), (inner_top_radius, rim_base_z)], includeCurrent=True)
    .lineTo(inner_top_radius, target_height + 3).lineTo(0, target_height + 3)
    .close().revolve(360, (0, 0), (0, 1)))
r = cup_shell.union(foot).union(handle).cut(NON_PENETRATION_CAVITY).cut(NON_PENETRATION_HANDLE_APERTURE).clean()
