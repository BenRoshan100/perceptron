from utils.model import Perceptron
from utils.all_utils import prepare_data, save_plot, save_model
import pandas as pd
import numpy as np
def main(data,eta,epochs,filename,plotfilename):
    
    df=pd.DataFrame(data)
    print(df)
    X,y=prepare_data(df)


    model=Perceptron(eta=eta,epochs=epochs)
    model.fit(X,y)

    _=model.total_loss()

    save_model(model,filename=filename)
    save_plot(df,filename=plotfilename,model=model)

if __name__=="main":

    AND = {
        "x1": [0,0,1,1],
        "x2": [0,1,0,1],
        "y": [0,0,0,1],
    }
    ETA=0.3
    EPOCHS=10
    main(data=AND,eta=ETA,epochs=EPOCHS,filename="and_model",plotfilename="and.png")