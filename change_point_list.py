import os

vocal_file_paths_and_change_points = [
    {
        "vocals":"../../ExpressionAnalysisVocal/data/Source Separated Vocals/all_of_me.wav",
        "change_points":[56.72, 59.59, 62.41, 65.33, 76.72, 81.5, 148.14, 151.04, 153.86, 156.75, 168.15, \
                         175.72, 198.42, 205.99, 209.86, 218.36, 230.06, 236.7],
        "dynamics_values":[5, 6, 5, 6, 5, 6, 5, 6, 5, 6, 5, 6, 5, 6, 5, 6, 5, 6, 5]
    },
    {
        "vocals":"../../ExpressionAnalysisVocal/data/Source Separated Vocals/chandelier.wav",
#        "change_points":[43.08, 44.45, 45.83, 109.28, 110.66, 112.03, 131.37, 132.73, 134.1]},
        "change_points": [12.04, 44.45, 110.66, 132.73],
        "dynamics_values": [5, 6, 5, 6]
    },
    {
        "vocals":"../../ExpressionAnalysisVocal/data/Source Separated Vocals/love_yourself.wav",
#        "change_points":[9.15, 10.36, 11.59, 162.76, 164.0, 165.2]
        "change_points":[10.36, 164.0],
        "dynamics_values":[5, 3]
    },
    {
        "vocals":"../../ExpressionAnalysisVocal/data/Source Separated Vocals/one_call_away.wav",
        "change_points":[64.95, 67.59, 79.45, 99.23, 127.58, 193.52, 213.31, 215.96],
        "dynamics_values":[]
    },
    {
        "vocals":"../../ExpressionAnalysisVocal/data/Source Separated Vocals/say_you_wont_let_go.wav",
        "change_points":[31.18, 51.39, 71.6, 82.35, 101.92, 122.13, 141.7, 153.71, 162.54, 182.12, 184.65,\
                         187.18, 194.76, 197.28, 199.81],
        "dynamics_values": [5, 6, 7, 6, 5, 6, 7, 6, 5, 7, 6, 5, 6, 5, 6, 5]
    },
    {
        "vocals":"../../ExpressionAnalysisVocal/data/Source Separated Vocals/when_i_was_your_man.wav",
        "change_points":[51.02, 57.59, 64.17, 69.1, 70.75, 75.66, 83.9, 108.56, 111.84,\
                        120.06, 125.81, 133.21, 138.14, 139.79, 144.71, 152.94, 158.7, 166.09,\
                        171.02, 178.43, 180.89, 183.37, 185.0, 190.9],
        "dynamics_values": [5, 3, 6, 5, 6, 5, 6, 5, 6, 5, 3, 6, 5, 6, 5, 6, 5, 6, 7, 6, 7, 6, 5, 3, 5]


    },
    {
        "vocals":"../../ExpressionAnalysisVocal/data/Source Separated Vocals/more_than_words.wav",
        "change_points":[29.53, 58.78, 72.35, 76.18, 85.34, 97.1, 107.57, 112.71, 113.36, 117.95,\
                        129.91, 164.31, 166.9, 174.8, 194.56, 197.12, 208.61, 219.12, 229.04, 234.87],
        "dynamics_values":[5]
    },
    {
        "vocals":"../../ExpressionAnalysisVocal/data/Source Separated Vocals/one_call_away.wav",
        #"change_points":[64.95, 67.59, 79.45, 99.23, 127.58, 193.52, 213.31, 215.96]
        "change_points":[64.95, 67.59, 81.44, 86.71, 101.87, 104.5, 112.42, 122.97, 124.97,\
                         127.58, 144.07, 151.32, 153.3, 156.6, 157.92, 161.87, 163.85,\
                         166.48, 168.47, 169.78, 193.52, 195.5, 197.46, 207.36, 208.68, 215.96]
    },
    {
        "vocals":"../../ExpressionAnalysisVocal/data/Source Separated Vocals/when_i_am_gone.wav",
        "change_points":[94.72, 100.72, 109.5, 116.88, 124.73, 129.34, 130.73, 151.03, 160.27,\
                         166.26, 203.18, 216.11]
    },
    {
        "vocals":"../../ExpressionAnalysisVocal/data/Source Separated Vocals/rockabye.wav",
        "change_points":[18.35, 56.56, 113.63, 131.86, 170.1, 200.59]
    },
    {
        "vocals":"../../ExpressionAnalysisVocal/data/Source Separated Vocals/despacito.wav",
        "change_points":[18.35, 56.56, 113.63, 131.86, 170.1, 200.59],
        "dynamics_values":[]
    },
    {
        "vocals":"../../ExpressionAnalysisVocal/data/Source Separated Vocals/seven_years.wav",
        "change_points":[21.08, 126.18, 133.2, 149.74, 190.24, 213.26],
        "dynamics_values":[3, 5, 2, 3, 5, 2]
    },
    {
        "vocals":"../../ExpressionAnalysisVocal/data/Source Separated Vocals/closer.wav",
        "change_points":[31.53, 91.53, 112.37, 172.38],
        "dynamics_values":[4,5,4,5,4]

    }
        
]

def get_change_points_for_all_songs():
    change_point_list = {}
    for song in vocal_file_paths_and_change_points:
        song_name = os.path.basename(song["vocals"])
        change_points = song["change_points"]
        change_point_list[song_name] = change_points
    return change_point_list   