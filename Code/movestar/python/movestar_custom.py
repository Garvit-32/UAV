from movestar import movestar
import pandas as pd
import csv
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv(
    "DJI_0004_gtdeep_sort.csv"
)

decoder = {"car": 1, "truck": 1, "bus": 2, "heavy_truck": 2}
speed_id_mapping = {}

mov_list = []
for i in data.iloc[:, :].values:
    id_out = movestar(decoder[i[7]], [i[8]])
    mov_list.append(list(i) + id_out["Emission Rate"]
                    [0] + id_out["Emission Factor"][0])

    # if i[2] in speed_id_mapping.keys():
    #     speed_id_mapping[i[2]]["speed"].append(i[8])
    # else:
    #     speed_id_mapping.update({i[2]: {"speed": [i[8]], "vehicle_id": i[7]}})

# for i in list(speed_id_mapping.keys()):
#     id_out = movestar(
#         speed_id_mapping[i]["vehicle_id"], speed_id_mapping[i]["speed"])
#     mov_list.append(list(i) + id_out["Emission Rate"]
#                     [0] + id_out["Emission Factor"][0])

with open("./merged1.csv", "w") as csvfile:
    # creating a csv writer object
    csvwriter = csv.writer(csvfile)

    # writing the fields
    csvwriter.writerow(
        [
            "frame id",
            "tracking id main",
            "tracking id class wise",
            "centroid x",
            "centroid y",
            "localized centroid x",
            "localized centroid y",
            "category",
            "speed",
            "acceleration",
            "average speed",
            "average acceleration",
            "CO(g/mi)",
            "HC(g/mi)",
            "NOx(g/mi)",
            "PM2.5_Ele(g/mi)",
            "PM2.5_Org(g/mi)",
            "Energy(KJ/mi)",
            "CO2(g/mi)",
            "Fuel(g/mi)",
            "TD(mi)",
            "CO(g)",
            "HC(g)",
            "NOx(g)",
            "PM2.5_Ele(g)",
            "PM2.5_Org(g)",
            "Energy(KJ)",
            "CO2(g)",
            "Fuel(g)",
            "TT(s)",
        ]
    )

    # writing the data rows
    csvwriter.writerows(mov_list)

# emission_rate = []
# emission_factor = []

# for i in list(speed_id_mapping.keys()):

#     id_out = movestar(
#         speed_id_mapping[i]["vehicle_id"], speed_id_mapping[i]["speed"])

#     emission_rate.append(
#         [i] + [speed_id_mapping[i]["vehicle_id"]] + id_out["Emission Rate"][0]
#     )
#     emission_factor.append(
#         [i] + [speed_id_mapping[i]["vehicle_id"]] + id_out["Emission Factor"][0]
#     )

# with open("./emission_rate.csv", "w") as csvfile:
#     # creating a csv writer object
#     csvwriter = csv.writer(csvfile)

#     # writing the fields
#     csvwriter.writerow(
#         [
#             "vehicle id",
#             "vehicle class",
#             "CO(g/mi)",
#             "HC(g/mi)",
#             "NOx(g/mi)",
#             "PM2.5_Ele(g/mi)",
#             "PM2.5_Org(g/mi)",
#             "Energy(KJ/mi)",
#             "CO2(g/mi)",
#             "Fuel(g/mi)",
#             "TD(mi)",
#         ]
#     )

#     # writing the data rows
#     csvwriter.writerows(emission_rate)

# with open("./emission_factor.csv", "w") as csvfile:
#     # creating a csv writer object
#     csvwriter = csv.writer(csvfile)

#     # writing the fields
#     csvwriter.writerow(
#         [
#             "vehicle id",
#             "vehicle class",
#             "CO(g)",
#             "HC(g)",
#             "NOx(g)",
#             "PM2.5_Ele(g)",
#             "PM2.5_Org(g)",
#             "Energy(KJ)",
#             "CO2(g)",
#             "Fuel(g)",
#             "TT(s)",
#         ]
#     )

#     # writing the data rows
#     csvwriter.writerows(emission_factor)
