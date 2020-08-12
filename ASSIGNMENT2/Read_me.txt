Both test.py and train.py can be run on Command line

On Command  Line set path to Our Assignment folder.

On Python Console :
 Run as 
1)  python train.py

Intially feature matrix is extracted from the data and PCA is performed  and TOP 5 Components are extracted 

Next Training is perfomed Using Gaussian Classifier and the Model is saved as Guassian_model.pkl

Output : Shows Precision,Recall, Accuracy and F1 Score on the existing data

2)Save your test data in our folder.
example : test_file.csv

Now run on Command prompt as
python test.py test_file.csv 

This will print the class lables and also save class lables in output.csv


