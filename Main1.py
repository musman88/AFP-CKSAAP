from models1 import *
from utilis1 import *
import matplotlib.pyplot as plt
import numpy as np
## DATASET 1
train_dist1 = (300,900)
# AAC_AFP = 'input/AFP_AAC.txt'
# AAC_NONAFP = 'input/Non-AFP_AAC.txt'
# CKS_AFP = 'input/AFP_CKSAAP.txt'
# CKS_NONAFP = 'input/Non-AFP_CKSAAP.txt'
# AFP1_path = np.concatenate((AAC_AFP, CKS_AFP), axis=1)
# NON_AFP1_path = np.concatenate((AAC_NONAFP, CKS_NONAFP),axis=1)

AFP1_path = 'input/AFP_CKSAAP13.txt'
NON_AFP1_path = 'input/Non-AFP_CKSAAP13.txt'
X_train1, Y_train1, X_test1, Y_test1, test_dist1, input_dim, nb_classes \
    = Generate_Datasets(AFP1_path,NON_AFP1_path,train_dist1)

## USE THIS if train_dist contain one or more zeros
# X_test1, Y_test1, test_dist1, input_dim, nb_classes \
#     = Generate_Datasets(AFP1_path,NON_AFP1_path,train_dist1)

## DATASET 2
# train_dist2 = (38,3255)
# AFP2_path = 'input/AFP2_CTDC.txt'
# NON_AFP2_path = 'input/Non-AFP2_CTDC.txt'
# # # USE THIS if train_dist contain one or more zeros
# # X_test2, Y_test2, test_dist2, input_dim, nb_classes \
# X_train2, Y_train2, X_test2, Y_test2, test_dist2, input_dim, nb_classes \
#    = Generate_Datasets(AFP2_path,NON_AFP2_path,train_dist2)

## DATASET 3
# train_dist3 = (38,3255)
# AFP3_path = 'input/AFP3i_CTDC.txt'
# NON_AFP3_path = 'input/Non-AFP2_CTDC.txt'
# # # USE THIS if train_dist contain one or more zeros
# # X_test2, Y_test2, test_dist2, input_dim, nb_classes \
# X_train2, Y_train2, X_test2, Y_test2, test_dist2, input_dim, nb_classes \
#    = Generate_Datasets(AFP2_path,NON_AFP2_path,train_dist2)

## Simple Dense Model
# model = RAFP_model(input_dim=input_dim, nb_classes=nb_classes)
# from Dense_with_Skip_Connections import *
model = RAFP_model_Skip(input_dim=input_dim, nb_classes=nb_classes)

print("Training...")
# For Dataset 1
history = model.fit(X_train1, Y_train1, epochs=200, batch_size=600, validation_split=0.1, verbose=2) # verbose = 2 for less details

# For Dataset 2
# history = model.fit(X_train2, Y_train2, epochs=1000, batch_size=600, validation_split=0.1, verbose=2) # verbose = 2 for less details

# For Dataset 3
# history = model.fit(X_train2, Y_train2, epochs=1000, batch_size=600, validation_split=0.1, verbose=2) # verbose = 2 for less details


print("Generating test predictions...")

print("FOR DATASET # 1")
Y_pred1 = model.predict(X_test1, verbose=0)
GenerateScore(Y_pred1,Y_test1)

# print("FOR DATASET # 2")
# Y_pred2 = model.predict(X_test2, verbose=0)
# GenerateScore(Y_pred2,Y_test2)

# print("FOR DATASET # 3")
# Y_pred3 = model.predict(X_test3, verbose=0)
# GenerateScore(Y_pred3,Y_test3)

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper right')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()