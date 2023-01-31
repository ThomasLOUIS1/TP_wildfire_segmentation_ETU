import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import PIL
from PIL import Image

def display_confusion_matrix(Y_pred, Y_true):

    FP = len(np.where(Y_pred - Y_true  == 1)[0])
    FN = len(np.where(Y_pred - Y_true  == -1)[0])
    TP = len(np.where(Y_pred + Y_true ==2)[0])
    TN = len(np.where(Y_pred + Y_true == 0)[0])
    cmat = [[TP, FN], [FP, TN]]

    plt.figure(figsize = (6,6))
    sns.heatmap(cmat/np.sum(cmat), cmap="Reds", annot=True, fmt = '.2%', square=1,   linewidth=2.)
    plt.xlabel("predictions")
    plt.ylabel("real values")
    plt.show()


def load_masks(paths = None, img_size = None):
    y = np.zeros((len(paths),) + img_size + (1,), dtype="float32")
    for i, (targ_path)  in enumerate(paths):
        out = np.array(Image.open(targ_path))
        out = np.expand_dims(out, 2)
        y[i] = out
    return y

def print_score(score) :
    score_names = ["loss", "Accuracy", "Recall", "Precision", "f1_score"]
    
    score_and_names = zip(score_names,score)
    
    for i,j in score_and_names :
        print("|{} : {}".format(i,j), end='|')

def display_matrix(matrix,title_list,path_save="image"):
    columns = len(matrix[0])
    rows = len(matrix)
    fig=plt.figure(figsize=(5*columns,5*rows))


    for j in range(0,rows):
        for i in range(0,columns):
            ax=fig.add_subplot(rows,columns,(columns*j)+i+1)
            if j == 0:
                ax.set_title(title_list[i],c='black')
            ax.set_aspect('equal')
            plt.imshow(matrix[j][i])
            plt.axis('off')
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.show()
    #fig.savefig(path_save+'.jpg',bbox_inches='tight', dpi=150)
    
def transform_predicted_mask(pred_mask, threshhold=0.5):
    pred_mask[pred_mask >= threshhold] = 1.0
    pred_mask[pred_mask < threshhold] = 0.0
    return pred_mask.astype(np.uint8)


def predict(model, image, threshhold=0.5):
    """
    use the model to predict a mask from image as input
    """
    pred_mask = model.predict(image)
    return transform_predicted_mask(pred_mask, threshhold=threshhold)


def write_triband_and_target_from_paths(paths = None, folder = None):
    import shutil
    for i, (in_path, targ_path)  in enumerate(paths):
        shutil.copy(in_path, "/hdd-raid0/home_server/thomasl/Vacations/MCIA/Wildfire_segmentation/data/false_color_new/"+folder)
        shutil.copy(targ_path, "/hdd-raid0/home_server/thomasl/Vacations/MCIA/Wildfire_segmentation/data/masks_new/"+folder)



# Used to create the TP

'''
# TO DO Choice the right proportion for the TP
import random

train_split = 0.3
val_split = 0.1
test_split = 0.1

combined = list(zip(triband_img_paths, target_img_paths))

random.shuffle(combined)

data_paths, label_paths = zip(*combined)

for input_path, target_path in zip(data_paths[:10], label_paths[:10]):
    print(input_path, "|", target_path)


train_index = int(len(combined) * train_split)
val_index = train_index + int(len(combined) * val_split)
test_index = val_index + int(len(combined) * test_split)

train_paths = combined[:train_index]
val_paths = combined[train_index:val_index]
test_paths = combined[val_index:test_index]

for paths in [train_data, val_data, test_data] :
  for input_path, target_path in paths[:10]:
      print(input_path, "|", target_path)
  print ("*"*30)

print("Train size : {} | Validation size : {} | Test size : {} ".format(train_index, val_index, test_index))

#############################################################################
# TEST ! DON NOT COMMIT TO GIT
#############################################################################

from tensorflow.keras import backend as K

from metrics_and_losses import recall_m, precision_m, f1_m
zeros_weight = 1.0

print("Classic bin cross")

model2 = None

model2 = Sequential()

model2.add(Conv2D(8, (3, 3), activation='relu', input_shape=(256, 256, 3), kernel_initializer=kernel_init, bias_initializer=bias_init))
model2.add(Conv2D(4, (3, 3), activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init))

model2.add(Conv2DTranspose(4, (3, 3), activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init))
model2.add(Conv2DTranspose(8, (3, 3), activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init))
model2.add(Conv2D(1, (3, 3), activation='sigmoid', padding='same'))

metrics = [tf.keras.metrics.BinaryAccuracy(), recall_m, precision_m, f1_m]

model2.compile(optimizer = Adam(learning_rate=0.01), metrics = metrics, loss = "binary_crossentropy")

history = model2.fit(train_ds, epochs = 4, validation_data = val_ds, batch_size=batch_size, verbose=0)
model2_score = model2.evaluate(test_ds)

for ones_weight in range(1 , 250, 25):

    model2 = None

    model2 = Sequential()

    model2.add(Conv2D(8, (3, 3), activation='relu', input_shape=(256, 256, 3), kernel_initializer=kernel_init, bias_initializer=bias_init))
    model2.add(Conv2D(4, (3, 3), activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init))

    model2.add(Conv2DTranspose(4, (3, 3), activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init))
    model2.add(Conv2DTranspose(8, (3, 3), activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init))
    model2.add(Conv2D(1, (3, 3), activation='sigmoid', padding='same'))

    #model2.summary()
    ones_weight = float(ones_weight)
    print(ones_weight)

    def weighted_binary_crossentropy( y_true, y_pred) :
        y_true = K.clip(y_true, K.epsilon(), 1-K.epsilon())
        y_pred = K.clip(y_pred, K.epsilon(), 1-K.epsilon())
        logloss = -(y_true * K.log(y_pred) * ones_weight + (1 - y_true) * K.log(1 - y_pred) * zeros_weight )
        return K.mean( logloss, axis=-1)

    metrics = [tf.keras.metrics.BinaryAccuracy(), recall_m, precision_m, f1_m]

    model2.compile(optimizer = Adam(learning_rate=0.01), metrics = metrics, loss = weighted_binary_crossentropy)

    history = model2.fit(train_ds, epochs = 4, validation_data = val_ds, batch_size=batch_size, verbose=0)
    model2_score = model2.evaluate(test_ds)
    #print_score(model2_score)
#################################################################################

'''