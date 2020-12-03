import numpy as np
from collections import defaultdict

AID_scenes = ['Airport', 'BareLand', 'BaseballField', 'Beach', 'Bridge', 'Center', 'Church', 'Commercial', 'DenseResidential', 'Desert', 'Farmland', 'Forest', 'Industrial', 'Meadow', 'MediumResidential', 'Mountain', 'Park', 'Parking', 'Playground', 'Pond', 'Port', 'RailwayStation', 'Resort', 'River', 'School', 'SparseResidential', 'Square', 'Stadium', 'StorageTanks', 'Viaduct']

nb_classes = 30

lb_Trans_05 = defaultdict()
lb_Trans_05['Airport'] = {'Airport':0.5, 'BareLand':0.1, 'Industrial':0.1, 'Parking':0.1, 'RailwayStation':0.1, 'StorageTanks':0.1}
lb_Trans_05['BareLand'] = {'BareLand':0.5, 'Desert':0.3, 'Mountain':0.2}
lb_Trans_05['BaseballField'] = {'BaseballField':0.5, 'Meadow':0.2, 'SparseResidential':0.1, 'Farmland':0.2}
lb_Trans_05['Beach'] = {'Beach':0.5, 'Bridge':0.1, 'Pond':0.2, 'Port':0.2}
lb_Trans_05['Bridge'] = {'Bridge':0.5, 'Beach':0.1, 'Pond':0.2, 'Port':0.2}
lb_Trans_05['Center'] = {'Center':0.5, 'Church':0.3, 'Commercial':0.1, 'Square':0.1}
lb_Trans_05['Church'] = {'Church':0.5, 'Center':0.3, 'Commercial':0.1, 'Square':0.1}
lb_Trans_05['Commercial'] = {'Commercial':0.5, 'Center':0.2, 'Church':0.1, 'Square':0.1, 'DenseResidential':0.1}
lb_Trans_05['DenseResidential'] = {'DenseResidential':0.5, 'Center':0.2, 'Church':0.1, 'Square':0.1, 'Resort':0.1}
lb_Trans_05['Desert'] = {'Desert':0.5, 'BareLand':0.3, 'Mountain':0.2}
lb_Trans_05['Farmland'] = {'Farmland':0.5, 'BaseballField':0.2, 'Forest':0.2, 'Meadow':0.1}
lb_Trans_05['Forest'] = {'Forest':0.5, 'BaseballField':0.2, 'Farmland':0.2, 'Meadow':0.1}
lb_Trans_05['Industrial'] = {'Industrial':0.5, 'Airport':0.1, 'Square':0.1, 'Viaduct':0.1, 'StorageTanks':0.2}
lb_Trans_05['Meadow'] = {'Meadow':0.5, 'BaseballField':0.1, 'Farmland':0.1, 'Forest':0.2, 'River':0.1}
lb_Trans_05['MediumResidential'] = {'MediumResidential':0.5, 'SparseResidential':0.3, 'DenseResidential':0.2}
lb_Trans_05['Mountain'] = {'Mountain':0.5, 'BareLand':0.2, 'Meadow':0.2, 'Farmland':0.1}
lb_Trans_05['Park'] = {'Park':0.5, 'School':0.2, 'SparseResidential':0.2, 'Square':0.1}
lb_Trans_05['Parking'] = {'Parking':0.5, 'Airport':0.2, 'RailwayStation':0.2, 'BareLand':0.1}
lb_Trans_05['Playground'] = {'Playground':0.5, 'Stadium':0.4, 'Park':0.1}
lb_Trans_05['Pond'] = {'Pond':0.5, 'Bridge':0.2, 'Port':0.1, 'River':0.2}
lb_Trans_05['Port'] = {'Port':0.5, 'Pond':0.2, 'Beach':0.1, 'River':0.2}
lb_Trans_05['Resort'] = {'Resort':0.5, 'Center':0.2, 'Church':0.2, 'Square':0.1}
lb_Trans_05['River'] = {'River':0.5, 'Bridge':0.2, 'Forest':0.1, 'Meadow':0.1, 'Pond':0.1}
lb_Trans_05['School'] = {'School':0.5, 'Resort':0.2, 'Stadium':0.1, 'Square':0.1, 'Center':0.1}
lb_Trans_05['SparseResidential'] = {'SparseResidential':0.5, 'MediumResidential':0.3, 'DenseResidential':0.2}
lb_Trans_05['Square'] = {'Square':0.5, 'Viaduct':0.2, 'Airport':0.2, 'Parking':0.1}
lb_Trans_05['Stadium'] = {'Stadium':0.5, 'Playground':0.4, 'School':0.1}
lb_Trans_05['StorageTanks'] = {'StorageTanks':0.5, 'Industrial':0.2, 'Airport':0.1, 'Parking':0.2}
lb_Trans_05['Viaduct'] = {'Viaduct':0.5, 'Square':0.3, 'RailwayStation':0.2}
lb_Trans_05['RailwayStation'] = {'RailwayStation':0.5, 'Square':0.3, 'Viaduct':0.2}


P_05 = np.zeros((nb_classes, nb_classes))

for k,v in lb_Trans_05.items():
#     print(list(v.values()), np.array(list(v.values())).sum())
#     assert np.array(list(v.values())).sum() == 1.0
    for k_, v_ in v.items():
        P_05[AID_scenes.index(k)][AID_scenes.index(k_)] = v_


##### noise_rate 0.1

lb_Trans_01 = defaultdict()
other_P = 0.1
lb_Trans_01['Airport'] = {'Airport':0.9, 'BareLand':other_P/5, 'Industrial':other_P/5, 'Parking':other_P/5, 'RailwayStation':other_P/5, 'StorageTanks':other_P/5}
lb_Trans_01['BareLand'] = {'BareLand':0.9, 'Desert':other_P/5*3, 'Mountain':other_P/5*2}
lb_Trans_01['BaseballField'] = {'BaseballField':0.9, 'Meadow':other_P/5*2, 'SparseResidential':other_P/5*1, 'Farmland':other_P/5*2}
lb_Trans_01['Beach'] = {'Beach':0.9, 'Bridge':other_P/5*1, 'Pond':other_P/5*2, 'Port':other_P/5*2}
lb_Trans_01['Bridge'] = {'Bridge':0.9, 'Beach':other_P/5, 'Pond':other_P/5*2, 'Port':other_P/5*2}
lb_Trans_01['Center'] = {'Center':0.9, 'Church':other_P/5*3, 'Commercial':other_P/5, 'Square':other_P/5}
lb_Trans_01['Church'] = {'Church':0.9, 'Center':other_P/5*3, 'Commercial':other_P/5, 'Square':other_P/5}
lb_Trans_01['Commercial'] = {'Commercial':0.9, 'Center':other_P/5*2, 'Church':other_P/5, 'Square':other_P/5, 'DenseResidential':other_P/5}
lb_Trans_01['DenseResidential'] = {'DenseResidential':0.9, 'Center':other_P/5*2, 'Church':other_P/5, 'Square':other_P/5, 'Resort':other_P/5}
lb_Trans_01['Desert'] = {'Desert':0.9, 'BareLand':other_P/5*3, 'Mountain':other_P/5*2}
lb_Trans_01['Farmland'] = {'Farmland':0.9, 'BaseballField':other_P/5*2, 'Forest':other_P/5*2, 'Meadow':other_P/5}
lb_Trans_01['Forest'] = {'Forest':0.9, 'BaseballField':other_P/5*2, 'Farmland':other_P/5*2, 'Meadow':other_P/5}
lb_Trans_01['Industrial'] = {'Industrial':0.9, 'Airport':other_P/5, 'Square':other_P/5, 'Viaduct':other_P/5, 'StorageTanks':other_P/5*2}
lb_Trans_01['Meadow'] = {'Meadow':0.9, 'BaseballField':other_P/5, 'Farmland':other_P/5, 'Forest':other_P/5*2, 'River':other_P/5}
lb_Trans_01['MediumResidential'] = {'MediumResidential':0.9, 'SparseResidential':other_P/5*3, 'DenseResidential':other_P/5*2}
lb_Trans_01['Mountain'] = {'Mountain':0.9, 'BareLand':other_P/5*2, 'Meadow':other_P/5*2, 'Farmland':other_P/5}
lb_Trans_01['Park'] = {'Park':0.9, 'School':other_P/5*2, 'SparseResidential':other_P/5*2, 'Square':other_P/5}
lb_Trans_01['Parking'] = {'Parking':0.9, 'Airport':other_P/5*2, 'RailwayStation':other_P/5*2, 'BareLand':other_P/5}
lb_Trans_01['Playground'] = {'Playground':0.9, 'Stadium':other_P/5*4, 'Park':other_P/5}
lb_Trans_01['Pond'] = {'Pond':0.9, 'Bridge':other_P/5*2, 'Port':other_P/5, 'River':other_P/5*2}
lb_Trans_01['Port'] = {'Port':0.9, 'Pond':other_P/5*2, 'Beach':other_P/5, 'River':other_P/5*2}
lb_Trans_01['Resort'] = {'Resort':0.9, 'Center':other_P/5*2, 'Church':other_P/5*2, 'Square':other_P/5}
lb_Trans_01['River'] = {'River':0.9, 'Bridge':other_P/5*2, 'Forest':other_P/5, 'Meadow':other_P/5, 'Pond':other_P/5}
lb_Trans_01['School'] = {'School':0.9, 'Resort':other_P/5*2, 'Stadium':other_P/5, 'Square':other_P/5, 'Center':other_P/5}
lb_Trans_01['SparseResidential'] = {'SparseResidential':0.9, 'MediumResidential':other_P/5*3, 'DenseResidential':other_P/5*2}
lb_Trans_01['Square'] = {'Square':0.9, 'Viaduct':other_P/5*2, 'Airport':other_P/5*2, 'Parking':other_P/5}
lb_Trans_01['Stadium'] = {'Stadium':0.9, 'Playground':other_P/5*4, 'School':other_P/5}
lb_Trans_01['StorageTanks'] = {'StorageTanks':0.9, 'Industrial':other_P/5*2, 'Airport':other_P/5, 'Parking':other_P/5*2}
lb_Trans_01['Viaduct'] = {'Viaduct':0.9, 'Square':other_P/5*3, 'RailwayStation':other_P/5*2}
lb_Trans_01['RailwayStation'] = {'RailwayStation':0.9, 'Square':other_P/5*3, 'Viaduct':other_P/5*2}

P_01 = np.zeros((nb_classes, nb_classes))

for k,v in lb_Trans_01.items():
#     print(list(v.values()), np.array(list(v.values())).sum())
#     assert np.array(list(v.values())).sum() == 1.0
    for k_, v_ in v.items():
        P_01[AID_scenes.index(k)][AID_scenes.index(k_)] = v_

# print(P_01.sum(axis=-1))

##### noise level 0.3

lb_Trans_03 = defaultdict()
other_P = 0.3
lb_Trans_03['Airport'] = {'Airport':0.7, 'BareLand':other_P/5, 'Industrial':other_P/5, 'Parking':other_P/5, 'RailwayStation':other_P/5, 'StorageTanks':other_P/5}
lb_Trans_03['BareLand'] = {'BareLand':0.7, 'Desert':other_P/5*3, 'Mountain':other_P/5*2}
lb_Trans_03['BaseballField'] = {'BaseballField':0.7, 'Meadow':other_P/5*2, 'SparseResidential':other_P/5*1, 'Farmland':other_P/5*2}
lb_Trans_03['Beach'] = {'Beach':0.7, 'Bridge':other_P/5*1, 'Pond':other_P/5*2, 'Port':other_P/5*2}
lb_Trans_03['Bridge'] = {'Bridge':0.7, 'Beach':other_P/5, 'Pond':other_P/5*2, 'Port':other_P/5*2}
lb_Trans_03['Center'] = {'Center':0.7, 'Church':other_P/5*3, 'Commercial':other_P/5, 'Square':other_P/5}
lb_Trans_03['Church'] = {'Church':0.7, 'Center':other_P/5*3, 'Commercial':other_P/5, 'Square':other_P/5}
lb_Trans_03['Commercial'] = {'Commercial':0.7, 'Center':other_P/5*2, 'Church':other_P/5, 'Square':other_P/5, 'DenseResidential':other_P/5}
lb_Trans_03['DenseResidential'] = {'DenseResidential':0.7, 'Center':other_P/5*2, 'Church':other_P/5, 'Square':other_P/5, 'Resort':other_P/5}
lb_Trans_03['Desert'] = {'Desert':0.7, 'BareLand':other_P/5*3, 'Mountain':other_P/5*2}
lb_Trans_03['Farmland'] = {'Farmland':0.7, 'BaseballField':other_P/5*2, 'Forest':other_P/5*2, 'Meadow':other_P/5}
lb_Trans_03['Forest'] = {'Forest':0.7, 'BaseballField':other_P/5*2, 'Farmland':other_P/5*2, 'Meadow':other_P/5}
lb_Trans_03['Industrial'] = {'Industrial':0.7, 'Airport':other_P/5, 'Square':other_P/5, 'Viaduct':other_P/5, 'StorageTanks':other_P/5*2}
lb_Trans_03['Meadow'] = {'Meadow':0.7, 'BaseballField':other_P/5, 'Farmland':other_P/5, 'Forest':other_P/5*2, 'River':other_P/5}
lb_Trans_03['MediumResidential'] = {'MediumResidential':0.7, 'SparseResidential':other_P/5*3, 'DenseResidential':other_P/5*2}
lb_Trans_03['Mountain'] = {'Mountain':0.7, 'BareLand':other_P/5*2, 'Meadow':other_P/5*2, 'Farmland':other_P/5}
lb_Trans_03['Park'] = {'Park':0.7, 'School':other_P/5*2, 'SparseResidential':other_P/5*2, 'Square':other_P/5}
lb_Trans_03['Parking'] = {'Parking':0.7, 'Airport':other_P/5*2, 'RailwayStation':other_P/5*2, 'BareLand':other_P/5}
lb_Trans_03['Playground'] = {'Playground':0.7, 'Stadium':other_P/5*4, 'Park':other_P/5}
lb_Trans_03['Pond'] = {'Pond':0.7, 'Bridge':other_P/5*2, 'Port':other_P/5, 'River':other_P/5*2}
lb_Trans_03['Port'] = {'Port':0.7, 'Pond':other_P/5*2, 'Beach':other_P/5, 'River':other_P/5*2}
lb_Trans_03['Resort'] = {'Resort':0.7, 'Center':other_P/5*2, 'Church':other_P/5*2, 'Square':other_P/5}
lb_Trans_03['River'] = {'River':0.7, 'Bridge':other_P/5*2, 'Forest':other_P/5, 'Meadow':other_P/5, 'Pond':other_P/5}
lb_Trans_03['School'] = {'School':0.7, 'Resort':other_P/5*2, 'Stadium':other_P/5, 'Square':other_P/5, 'Center':other_P/5}
lb_Trans_03['SparseResidential'] = {'SparseResidential':0.7, 'MediumResidential':other_P/5*3, 'DenseResidential':other_P/5*2}
lb_Trans_03['Square'] = {'Square':0.7, 'Viaduct':other_P/5*2, 'Airport':other_P/5*2, 'Parking':other_P/5}
lb_Trans_03['Stadium'] = {'Stadium':0.7, 'Playground':other_P/5*4, 'School':other_P/5}
lb_Trans_03['StorageTanks'] = {'StorageTanks':0.7, 'Industrial':other_P/5*2, 'Airport':other_P/5, 'Parking':other_P/5*2}
lb_Trans_03['Viaduct'] = {'Viaduct':0.7, 'Square':other_P/5*3, 'RailwayStation':other_P/5*2}
lb_Trans_03['RailwayStation'] = {'RailwayStation':0.7, 'Square':other_P/5*3, 'Viaduct':other_P/5*2}

P_03 = np.zeros((nb_classes, nb_classes))

for k,v in lb_Trans_03.items():
#     print(list(v.values()), np.array(list(v.values())).sum())
#     assert np.array(list(v.values())).sum() == 1.0
    for k_, v_ in v.items():
        P_03[AID_scenes.index(k)][AID_scenes.index(k_)] = v_

# print(P_03.sum(axis=-1))


###### noise rate 0.7


lb_Trans_07 = defaultdict()
other_P = 0.7
lb_Trans_07['Airport'] = {'Airport':0.3, 'BareLand':other_P/5, 'Industrial':other_P/5, 'Parking':other_P/5, 'RailwayStation':other_P/5, 'StorageTanks':other_P/5}
lb_Trans_07['BareLand'] = {'BareLand':0.3, 'Desert':other_P/5*3, 'Mountain':other_P/5*2}
lb_Trans_07['BaseballField'] = {'BaseballField':0.3, 'Meadow':other_P/5*2, 'SparseResidential':other_P/5*1, 'Farmland':other_P/5*2}
lb_Trans_07['Beach'] = {'Beach':0.3, 'Bridge':other_P/5*1, 'Pond':other_P/5*2, 'Port':other_P/5*2}
lb_Trans_07['Bridge'] = {'Bridge':0.3, 'Beach':other_P/5, 'Pond':other_P/5*2, 'Port':other_P/5*2}
lb_Trans_07['Center'] = {'Center':0.3, 'Church':other_P/5*3, 'Commercial':other_P/5, 'Square':other_P/5}
lb_Trans_07['Church'] = {'Church':0.3, 'Center':other_P/5*3, 'Commercial':other_P/5, 'Square':other_P/5}
lb_Trans_07['Commercial'] = {'Commercial':0.3, 'Center':other_P/5*2, 'Church':other_P/5, 'Square':other_P/5, 'DenseResidential':other_P/5}
lb_Trans_07['DenseResidential'] = {'DenseResidential':0.3, 'Center':other_P/5*2, 'Church':other_P/5, 'Square':other_P/5, 'Resort':other_P/5}
lb_Trans_07['Desert'] = {'Desert':0.3, 'BareLand':other_P/5*3, 'Mountain':other_P/5*2}
lb_Trans_07['Farmland'] = {'Farmland':0.3, 'BaseballField':other_P/5*2, 'Forest':other_P/5*2, 'Meadow':other_P/5}
lb_Trans_07['Forest'] = {'Forest':0.3, 'BaseballField':other_P/5*2, 'Farmland':other_P/5*2, 'Meadow':other_P/5}
lb_Trans_07['Industrial'] = {'Industrial':0.3, 'Airport':other_P/5, 'Square':other_P/5, 'Viaduct':other_P/5, 'StorageTanks':other_P/5*2}
lb_Trans_07['Meadow'] = {'Meadow':0.3, 'BaseballField':other_P/5, 'Farmland':other_P/5, 'Forest':other_P/5*2, 'River':other_P/5}
lb_Trans_07['MediumResidential'] = {'MediumResidential':0.3, 'SparseResidential':other_P/5*3, 'DenseResidential':other_P/5*2}
lb_Trans_07['Mountain'] = {'Mountain':0.3, 'BareLand':other_P/5*2, 'Meadow':other_P/5*2, 'Farmland':other_P/5}
lb_Trans_07['Park'] = {'Park':0.3, 'School':other_P/5*2, 'SparseResidential':other_P/5*2, 'Square':other_P/5}
lb_Trans_07['Parking'] = {'Parking':0.3, 'Airport':other_P/5*2, 'RailwayStation':other_P/5*2, 'BareLand':other_P/5}
lb_Trans_07['Playground'] = {'Playground':0.3, 'Stadium':other_P/5*4, 'Park':other_P/5}
lb_Trans_07['Pond'] = {'Pond':0.3, 'Bridge':other_P/5*2, 'Port':other_P/5, 'River':other_P/5*2}
lb_Trans_07['Port'] = {'Port':0.3, 'Pond':other_P/5*2, 'Beach':other_P/5, 'River':other_P/5*2}
lb_Trans_07['Resort'] = {'Resort':0.3, 'Center':other_P/5*2, 'Church':other_P/5*2, 'Square':other_P/5}
lb_Trans_07['River'] = {'River':0.3, 'Bridge':other_P/5*2, 'Forest':other_P/5, 'Meadow':other_P/5, 'Pond':other_P/5}
lb_Trans_07['School'] = {'School':0.3, 'Resort':other_P/5*2, 'Stadium':other_P/5, 'Square':other_P/5, 'Center':other_P/5}
lb_Trans_07['SparseResidential'] = {'SparseResidential':0.3, 'MediumResidential':other_P/5*3, 'DenseResidential':other_P/5*2}
lb_Trans_07['Square'] = {'Square':0.3, 'Viaduct':other_P/5*2, 'Airport':other_P/5*2, 'Parking':other_P/5}
lb_Trans_07['Stadium'] = {'Stadium':0.3, 'Playground':other_P/5*4, 'School':other_P/5}
lb_Trans_07['StorageTanks'] = {'StorageTanks':0.3, 'Industrial':other_P/5*2, 'Airport':other_P/5, 'Parking':other_P/5*2}
lb_Trans_07['Viaduct'] = {'Viaduct':0.3, 'Square':other_P/5*3, 'RailwayStation':other_P/5*2}
lb_Trans_07['RailwayStation'] = {'RailwayStation':0.3, 'Square':other_P/5*3, 'Viaduct':other_P/5*2}

P_07 = np.zeros((nb_classes, nb_classes))

for k,v in lb_Trans_07.items():
#     print(list(v.values()), np.array(list(v.values())).sum())
#     assert np.array(list(v.values())).sum() == 1.0
    for k_, v_ in v.items():
        P_07[AID_scenes.index(k)][AID_scenes.index(k_)] = v_

# print(P_07.sum(axis=-1))