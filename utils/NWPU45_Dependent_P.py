import numpy as np
from collections import defaultdict

NWPU45_scenes = ['airplane', 'airport', 'baseball_diamond', 'basketball_court', 'beach', 'bridge', 'chaparral', 'church', 'circular_farmland', 'cloud', 'commercial_area', 'dense_residential', 'desert', 'forest', 'freeway', 'golf_course', 'ground_track_field', 'harbor', 'industrial_area', 'intersection', 'island', 'lake', 'meadow', 'medium_residential', 'mobile_home_park', 'mountain', 'overpass', 'palace', 'parking_lot', 'railway', 'railway_station', 'rectangular_farmland', 'river', 'roundabout', 'runway', 'sea_ice', 'ship', 'snowberg', 'sparse_residential', 'stadium', 'storage_tank', 'tennis_court', 'terrace', 'thermal_power_station', 'wetland']

nb_classes = 45

def LB_Trans_P(other_P):

    lb_Trans = defaultdict()
    lb_Trans['airplane'] = {'airplane':1-other_P, 'airport':other_P/5*2, 'freeway':other_P/5, 'industrial_area':other_P/5, 'runway':other_P/5}
    lb_Trans['airport'] = {'airport':1-other_P, 'airplane':other_P/5, 'freeway':other_P/5, 'industrial_area':other_P/5, 'runway':other_P/5, 'storage_tank':other_P/5}
    lb_Trans['baseball_diamond'] = {'baseball_diamond':1-other_P, 'basketball_court':other_P/5, 'circular_farmland':other_P/5, 'golf_course':other_P/5*2, 'rectangular_farmland':other_P/5}
    lb_Trans['basketball_court'] = {'basketball_court':1-other_P, 'baseball_diamond':other_P/5, 'ground_track_field':other_P/5*3, 'tennis_court':other_P/5}
    lb_Trans['beach'] = {'beach':1-other_P, 'bridge':other_P/5, 'harbor':other_P/5, 'island':other_P/5, 'lake':other_P/5, 'river':other_P/5}
    lb_Trans['bridge'] = {'bridge':1-other_P, 'beach':other_P/5, 'wetland':other_P/5, 'island':other_P/5, 'lake':other_P/5, 'river':other_P/5}
    lb_Trans['chaparral'] = {'chaparral':1-other_P, 'desert':other_P/5*4, 'terrace':other_P/5}
    lb_Trans['church'] = {'church':1-other_P, 'commercial_area':other_P/5*2, 'industrial_area':other_P/5, 'palace':other_P/5*2}
    lb_Trans['circular_farmland'] = {'circular_farmland':1-other_P, 'baseball_diamond':other_P/5, 'forest':other_P/5, 'meadow':other_P/5, 'rectangular_farmland':other_P/5*2}
    lb_Trans['cloud'] = {'cloud':1-other_P, 'island':other_P/5, 'sea_ice':other_P/5*2, 'snowberg':other_P/5*2}
    lb_Trans['commercial_area'] = {'commercial_area':1-other_P, 'church':other_P/5, 'dense_residential':other_P/5*2, 'industrial_area':other_P/5, 'roundabout':other_P/5}
    lb_Trans['dense_residential'] = {'dense_residential':1-other_P, 'mobile_home_park':other_P/5, 'commercial_area':other_P/5*2, 'thermal_power_station':other_P/5, 'intersection':other_P/5}
    lb_Trans['desert'] = {'desert':1-other_P, 'chaparral':other_P/5*3, 'terrace':other_P/5*2}
    lb_Trans['forest'] = {'forest':1-other_P, 'golf_course':other_P/5, 'meadow':other_P/5*2, 'wetland':other_P/5, 'rectangular_farmland':other_P/5}
    lb_Trans['freeway'] = {'freeway':1-other_P, 'bridge':other_P/5, 'intersection':other_P/5, 'overpass':other_P/5, 'railway':other_P/5, 'runway':other_P/5}
    lb_Trans['golf_course'] = {'golf_course':1-other_P, 'baseball_diamond':other_P/5, 'sparse_residential':other_P/5, 'forest':other_P/5, 'meadow':other_P/5*2}
    lb_Trans['ground_track_field'] = {'ground_track_field':1-other_P, 'tennis_court':other_P/5, 'stadium':other_P/5, 'basketball_court':other_P/5, 'baseball_diamond':other_P/5*2}
    lb_Trans['harbor'] = {'harbor':1-other_P, 'beach':other_P/5, 'island':other_P/5, 'mobile_home_park':other_P/5*2, 'ship':other_P/5}
    lb_Trans['industrial_area'] = {'industrial_area':1-other_P, 'storage_tank':other_P/5, 'thermal_power_station':other_P/5, 'railway_station':other_P/5*2, 'roundabout':other_P/5}
    lb_Trans['intersection'] = {'intersection':1-other_P, 'roundabout':other_P/5, 'railway_station':other_P/5, 'overpass':other_P/5*2, 'commercial_area':other_P/5}
    lb_Trans['island'] = {'island':1-other_P, 'beach':other_P/5*2, 'wetland':other_P/5, 'lake':other_P/5*2}
    lb_Trans['lake'] = {'lake':1-other_P, 'beach':other_P/5*2, 'wetland':other_P/5*2, 'island':other_P/5}
    lb_Trans['meadow'] = {'meadow':1-other_P, 'forest':other_P/5*2, 'wetland':other_P/5*3}
    lb_Trans['medium_residential'] = {'medium_residential':1-other_P, 'forest':other_P/5*2, 'ground_track_field':other_P/5, 'tennis_court':other_P/5, 'sparse_residential':other_P/5}
    lb_Trans['mobile_home_park'] = {'mobile_home_park':1-other_P, 'railway_station':other_P/5*2, 'commercial_area':other_P/5, 'dense_residential':other_P/5, 'industrial_area':other_P/5}
    lb_Trans['mountain'] = {'mountain':1-other_P, 'meadow':other_P/5*2, 'snowberg':other_P/5, 'terrace':other_P/5*2}
    lb_Trans['overpass'] = {'overpass':1-other_P, 'freeway':other_P/5*2, 'intersection':other_P/5, 'railway':other_P/5, 'runway':other_P/5}
    lb_Trans['palace'] = {'palace':1-other_P, 'church':other_P/5*3, 'commercial_area':other_P/5, 'dense_residential':other_P/5}
    lb_Trans['parking_lot'] = {'parking_lot':1-other_P, 'mobile_home_park':other_P/5*3, 'railway_station':other_P/5, 'runway':other_P/5}
    lb_Trans['railway'] = {'railway':1-other_P, 'freeway':other_P/5*2, 'intersection':other_P/5, 'runway':other_P/5, 'railway_station':other_P/5}
    lb_Trans['railway_station'] = {'railway_station':1-other_P, 'railway':other_P/5*2, 'industrial_area':other_P/5, 'airport':other_P/5, 'runway':other_P/5}
    lb_Trans['rectangular_farmland'] = {'rectangular_farmland':1-other_P, 'baseball_diamond':other_P/5*2, 'circular_farmland':other_P/5, 'meadow':other_P/5, 'wetland':other_P/5}
    lb_Trans['river'] = {'river':1-other_P, 'bridge':other_P/5, 'meadow':other_P/5, 'lake':other_P/5, 'wetland':other_P/5*2}
    lb_Trans['roundabout'] = {'roundabout':1-other_P, 'intersection':other_P/5*4, 'ground_track_field':other_P/5}
    lb_Trans['runway'] = {'runway':1-other_P, 'airport':other_P/5, 'freeway':other_P/5*2, 'intersection':other_P/5, 'overpass':other_P/5}
    lb_Trans['sea_ice'] = {'sea_ice':1-other_P, 'island':other_P/5, 'cloud':other_P/5*2, 'wetland':other_P/5, 'snowberg':other_P/5}
    lb_Trans['ship'] = {'ship':1-other_P, 'harbor':other_P/5*2, 'bridge':other_P/5*2, 'river':other_P/5}
    lb_Trans['snowberg'] = {'snowberg':1-other_P, 'cloud':other_P/5*3, 'sea_ice':other_P/5*2}
    lb_Trans['sparse_residential'] = {'sparse_residential':1-other_P, 'baseball_diamond':other_P/5*2, 'golf_course':other_P/5*2, 'meadow':other_P/5}
    lb_Trans['stadium'] = {'stadium':1-other_P, 'tennis_court':other_P/5*2, 'basketball_court':other_P/5*3}
    lb_Trans['storage_tank'] = {'storage_tank':1-other_P, 'industrial_area':other_P/5*4, 'mobile_home_park':other_P/5*1}
    lb_Trans['tennis_court'] = {'tennis_court':1-other_P, 'baseball_diamond':other_P/5*2, 'basketball_court':other_P/5, 'stadium':other_P/5*2}
    lb_Trans['terrace'] = {'terrace':1-other_P, 'circular_farmland':other_P/5*2, 'chaparral':other_P/5, 'rectangular_farmland':other_P/5*2}
    lb_Trans['thermal_power_station'] = {'thermal_power_station':1-other_P, 'industrial_area':other_P/5*2, 'cloud':other_P/5, 'storage_tank':other_P/5*2}
    lb_Trans['wetland'] = {'wetland':1-other_P, 'bridge':other_P/5, 'forest':other_P/5, 'lake':other_P/5*2, 'river':other_P/5}

    return lb_Trans



P_01 = np.zeros((nb_classes, nb_classes))
P_03 = np.zeros((nb_classes, nb_classes))
P_05 = np.zeros((nb_classes, nb_classes))
P_07 = np.zeros((nb_classes, nb_classes))

lb_Trans_01 = LB_Trans_P(0.1)
lb_Trans_03 = LB_Trans_P(0.3)
lb_Trans_05 = LB_Trans_P(0.5)
lb_Trans_07 = LB_Trans_P(0.7)



for k,v in lb_Trans_01.items():
#     print(list(v.values()), np.array(list(v.values())).sum())
#     assert np.array(list(v.values())).sum() == 1.0
    for k_, v_ in v.items():
        P_01[NWPU45_scenes.index(k)][NWPU45_scenes.index(k_)] = v_


for k,v in lb_Trans_03.items():
#     print(list(v.values()), np.array(list(v.values())).sum())
#     assert np.array(list(v.values())).sum() == 1.0
    for k_, v_ in v.items():
        P_03[NWPU45_scenes.index(k)][NWPU45_scenes.index(k_)] = v_


for k,v in lb_Trans_05.items():
#     print(list(v.values()), np.array(list(v.values())).sum())
#     assert np.array(list(v.values())).sum() == 1.0
    for k_, v_ in v.items():
        P_05[NWPU45_scenes.index(k)][NWPU45_scenes.index(k_)] = v_


for k,v in lb_Trans_07.items():
#     print(list(v.values()), np.array(list(v.values())).sum())
#     assert np.array(list(v.values())).sum() == 1.0
    for k_, v_ in v.items():
        P_07[NWPU45_scenes.index(k)][NWPU45_scenes.index(k_)] = v_


# print(P_01.sum(axis=-1), len(P_01.sum(axis=-1)))
# print(P_03.sum(axis=-1), len(P_03.sum(axis=-1)))
# print(P_05.sum(axis=-1), len(P_05.sum(axis=-1)))
# print(P_07.sum(axis=-1), len(P_07.sum(axis=-1)))





