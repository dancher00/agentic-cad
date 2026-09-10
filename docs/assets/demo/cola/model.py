PARAMETERS = {'target_height': 120.0, 'wall_thickness': 0.2, 'body_diameter': 75.6, 'lower_chime_height': 9.7, 'base_recess_height': 3.3, 'base_recess_flat_diameter': 43.0, 'base_contact_inner_diameter': 52.0, 'base_contact_outer_diameter': 57.0, 'base_contact_lip_diameter': 61.0, 'base_chime_broad_diameter': 73.0, 'base_chime_broad_height': 6.1, 'shoulder_start_height': 100.5, 'shoulder_mid_diameter': 74.0, 'shoulder_mid_height': 106.8, 'shoulder_upper_diameter': 70.6, 'shoulder_upper_height': 111.7, 'neck_diameter': 65.2, 'neck_start_height': 114.6, 'rim_axial_height': 4.0, 'rim_outer_diameter': 68.6, 'rim_inner_diameter': 63.0, 'rim_outer_top_round_drop': 1.2, 'lid_recess_below_top': 2.3}
import cadquery as cq
import da3_cad.cad.profiles as profiles
target_height = PARAMETERS['target_height']
wall_thickness = PARAMETERS['wall_thickness']
body_diameter = PARAMETERS['body_diameter']
lower_chime_height = PARAMETERS['lower_chime_height']
base_recess_height = PARAMETERS['base_recess_height']
base_recess_flat_diameter = PARAMETERS['base_recess_flat_diameter']
base_recess_slope_diameter = 49.0
base_recess_slope_height = 1.45
base_contact_inner_diameter = PARAMETERS['base_contact_inner_diameter']
base_contact_outer_diameter = PARAMETERS['base_contact_outer_diameter']
base_contact_lip_diameter = PARAMETERS['base_contact_lip_diameter']
base_contact_lip_height = 1.25
base_chime_mid_diameter = 67.0
base_chime_mid_height = 3.5
base_chime_broad_diameter = PARAMETERS['base_chime_broad_diameter']
base_chime_broad_height = PARAMETERS['base_chime_broad_height']
base_chime_upper_diameter = 75.2
base_chime_upper_height = 8.3
shoulder_start_height = PARAMETERS['shoulder_start_height']
shoulder_lower_diameter = 75.3
shoulder_lower_height = 102.4
shoulder_mid_diameter = PARAMETERS['shoulder_mid_diameter']
shoulder_mid_height = PARAMETERS['shoulder_mid_height']
shoulder_upper_diameter = PARAMETERS['shoulder_upper_diameter']
shoulder_upper_height = PARAMETERS['shoulder_upper_height']
neck_diameter = PARAMETERS['neck_diameter']
neck_start_height = PARAMETERS['neck_start_height']
rim_axial_height = PARAMETERS['rim_axial_height']
rim_junction_diameter = 66.2
rim_junction_rise = 0.55
rim_lower_face_outer_diameter = 66.8
rim_outer_diameter = PARAMETERS['rim_outer_diameter']
rim_outer_wall_bottom_rise = 1.75
rim_outer_top_round_drop = PARAMETERS['rim_outer_top_round_drop']
rim_top_outer_flat_diameter = 67.0
rim_top_inner_flat_diameter = 64.7
rim_inner_diameter = PARAMETERS['rim_inner_diameter']
rim_inner_wall_top_drop = 1.2
rim_inner_wall_bottom_above_lid = 0.5
rim_lid_transition_diameter = 61.6
lid_recess_below_top = PARAMETERS['lid_recess_below_top']
revolve_angle = 360.0
body_radius = body_diameter / 2.0
base_recess_flat_radius = base_recess_flat_diameter / 2.0
base_recess_slope_radius = base_recess_slope_diameter / 2.0
base_contact_inner_radius = base_contact_inner_diameter / 2.0
base_contact_outer_radius = base_contact_outer_diameter / 2.0
base_contact_lip_radius = base_contact_lip_diameter / 2.0
base_chime_mid_radius = base_chime_mid_diameter / 2.0
base_chime_broad_radius = base_chime_broad_diameter / 2.0
base_chime_upper_radius = base_chime_upper_diameter / 2.0
shoulder_lower_radius = shoulder_lower_diameter / 2.0
shoulder_mid_radius = shoulder_mid_diameter / 2.0
shoulder_upper_radius = shoulder_upper_diameter / 2.0
neck_radius = neck_diameter / 2.0
rim_bottom_height = target_height - rim_axial_height
rim_junction_radius = rim_junction_diameter / 2.0
rim_lower_face_height = rim_bottom_height + rim_junction_rise
rim_lower_face_outer_radius = rim_lower_face_outer_diameter / 2.0
rim_outer_radius = rim_outer_diameter / 2.0
rim_outer_wall_bottom_height = rim_bottom_height + rim_outer_wall_bottom_rise
rim_outer_wall_top_height = target_height - rim_outer_top_round_drop
rim_top_outer_flat_radius = rim_top_outer_flat_diameter / 2.0
rim_top_inner_flat_radius = rim_top_inner_flat_diameter / 2.0
rim_inner_radius = rim_inner_diameter / 2.0
rim_inner_wall_top_height = target_height - rim_inner_wall_top_drop
lid_surface_height = target_height - lid_recess_below_top
rim_inner_wall_bottom_height = lid_surface_height + rim_inner_wall_bottom_above_lid
rim_lid_transition_radius = rim_lid_transition_diameter / 2.0
section = cq.Workplane('XZ').moveTo(0.0, base_recess_height).lineTo(base_recess_flat_radius, base_recess_height)
section = profiles.curve(section, [(base_recess_slope_radius, base_recess_slope_height), (base_contact_inner_radius, 0.0)], start_tangent=(1.0, 0.0), end_tangent=(1.0, 0.0))
section = section.lineTo(base_contact_outer_radius, 0.0)
section = profiles.roundover(section, (base_contact_lip_radius, base_contact_lip_height), start_tangent=(1.0, 0.0), end_tangent=(1.0, 1.0))
section = profiles.curve(section, [(base_chime_mid_radius, base_chime_mid_height), (base_chime_broad_radius, base_chime_broad_height), (base_chime_upper_radius, base_chime_upper_height), (body_radius, lower_chime_height)], start_tangent=(1.0, 1.0), end_tangent=(0.0, 1.0))
section = section.lineTo(body_radius, shoulder_start_height)
section = profiles.curve(section, [(shoulder_lower_radius, shoulder_lower_height), (shoulder_mid_radius, shoulder_mid_height), (shoulder_upper_radius, shoulder_upper_height), (neck_radius, neck_start_height)], start_tangent=(0.0, 1.0), end_tangent=(0.0, 1.0))
section = section.lineTo(neck_radius, rim_bottom_height)
section = profiles.roundover(section, (rim_junction_radius, rim_lower_face_height), start_tangent=(0.0, 1.0), end_tangent=(1.0, 0.0))
section = section.lineTo(rim_lower_face_outer_radius, rim_lower_face_height)
section = profiles.roundover(section, (rim_outer_radius, rim_outer_wall_bottom_height), start_tangent=(1.0, 0.0), end_tangent=(0.0, 1.0))
section = section.lineTo(rim_outer_radius, rim_outer_wall_top_height)
section = profiles.roundover(section, (rim_top_outer_flat_radius, target_height), start_tangent=(0.0, 1.0), end_tangent=(-1.0, 0.0))
section = section.lineTo(rim_top_inner_flat_radius, target_height)
section = profiles.roundover(section, (rim_inner_radius, rim_inner_wall_top_height), start_tangent=(-1.0, 0.0), end_tangent=(0.0, -1.0))
section = section.lineTo(rim_inner_radius, rim_inner_wall_bottom_height)
section = profiles.roundover(section, (rim_lid_transition_radius, lid_surface_height), start_tangent=(0.0, -1.0), end_tangent=(-1.0, 0.0))
section = section.lineTo(0.0, lid_surface_height)
exterior = section.close().revolve(revolve_angle, (0.0, 0.0), (0.0, 1.0))
hollow = exterior.shell(-wall_thickness)
NON_PENETRATION_CAVITY = exterior.cut(hollow)
r = hollow
