import math
import os

maximum_tick = 10000

use_all_folders = True

row = 3
col = 4

folders = ["100_node",
           "200_node",
           "400_node", "800_node", "1600_node", "2400_node", "3200_node", "4800_node", "6400_node", "8000_node"

           ]


titles = ["100_node",
          "200_node",
          "400_node", "800_node", "1600_node", "2400_node", "3200_node", "4800_node", "6400_node", "8000_node"

          ]

if use_all_folders:
    exclusion_list = ['__pycache__', '.idea']
    main_folder_path = '.'
    items = os.listdir(main_folder_path)
    folders = [item for item in items if os.path.isdir(os.path.join(main_folder_path, item)) and item not in exclusion_list]
    print(folders)
    titles = folders
    row = math.ceil(math.sqrt(len(folders)))
    col = int(len(folders) / row) + 1
