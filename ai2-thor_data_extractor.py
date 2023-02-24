#!pip install --upgrade ai2thor ai2thor-colab pickl5 scikit-learn &> /dev/null
import ai2thor
import ai2thor_colab
import pickle
import os.path

from ai2thor.controller import Controller
from ai2thor_colab import (
    plot_frames,
    show_objects_table,
    side_by_side,
    overlay,
    show_video
)

from sklearn.utils import shuffle

#ai2thor_colab.start_xserver()

print("AI2-THOR Version: " + ai2thor.__version__)

# Now generate room names
kitchens = ["FloorPlan" + str(i + 1) for i in range(30)]
living_rooms = ["FloorPlan" + str(i + 201) for i in range(30)]
bedrooms = ["FloorPlan" + str(i + 301) for i in range(30)]
bathrooms = ["FloorPlan" + str(i + 401) for i in range(30)]

floor_plans = kitchens + living_rooms + bedrooms + bathrooms
labels = ["kitchen" for i in range(30)] + ["living room" for i in range(30)] + ["bedroom" for i in range(30)] + ["bathroom" for i in range(30)]

# shuffle them so that they come in random order for training
floor_plans_shuffled,labels_shuffled = shuffle(floor_plans, labels, random_state=1983)

#print(kitchens)
#print(living_rooms)
#print(bedrooms)
#print(bathrooms)
#print(features)
#print(labels)
print(floor_plans_shuffled)
print(labels_shuffled)

# store our labels into a pickle file
labels_fname = "labels_shuffled.pkl"
features_fname = "features_for_each_label.pkl"

pickle.dump(labels_shuffled, open(labels_fname, "wb"))

# Now launch AI2-Thor with different rooms, get the list of objects in them and store that data into a pickle file.
# This part (or the whole script) will have to be launched multiple times until we get all rooms done. It's done this
# way because AI2-Thor takes a lot of memory with each instance and we can easily run out of it if we run all rooms
# at once.

# If pickle file for our features exists, then load features from there and continue with what we already have, 
# otherwise create an empty collection
if os.path.isfile(features_fname):
    file = open(features_fname,'rb')
    features_for_each_label = pickle.load(file)
    file.close()
else:
    features_for_each_label = []

#for i in range(len(floor_plans_shuffled)):
for i in range(len(features_for_each_label), len(features_for_each_label) + 10):
    if i > 5: break
    controller = Controller(
        scene=floor_plans_shuffled[i]
    )
    #plot_frames(controller.last_event)    
    objects = controller.last_event.metadata['objects']
    objs_to_store = set() # we'll put our objects into a set to make sure that we don't repeat them more than once.
    
    for obj in objects:
      #if not obj['visible']:
      #print(obj['objectType']) 
        objs_to_store.add(obj['objectType'])

    # now we'll get the objects into a string separated by a space
    objs_in_room_as_string = ""
    for obj in objs_to_store:
        objs_in_room_as_string += obj + " "
        
    #print(str(i + 1) + ") " + labels_shuffled[i] + " : " + str(objs_to_store))
    print(str(i + 1) + ") " + labels_shuffled[i] + " : " + objs_in_room_as_string[:-1])
    features_for_each_label.append(objs_in_room_as_string[:-1])
    
# finally put what we've got into a pickle file
print(features_for_each_label)
pickle.dump(features_for_each_label, open(features_fname, "wb"))
