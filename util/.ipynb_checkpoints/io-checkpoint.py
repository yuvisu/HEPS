import os
import pickle

def check_saving_path(root_dir, output_dir,model_name,model_id):
    save_dir = os.path.join(root_dir, output_dir, model_name)
    isExist = os.path.exists(save_dir)
    if isExist is False: os.makedirs(save_dir)
    save_path = os.path.join(save_dir, model_id)
    return save_path

def save_model(model, root_dir, models_dir, model_name, model_id):
    save_path = check_saving_path(root_dir, models_dir,model_name,model_id)
    pickle.dump(model, open(save_path, 'wb'))
    
def load_model(root_dir, models_dir, model_name, model_id):
    save_dir = os.path.join(root_dir, models_dir, model_name)
    isExist = os.path.exists(save_dir)
    if isExist is False: 
        print("No folder exist!!!")
        return None
    save_path = os.path.join(save_dir, model_id)
    loaded_model = pickle.load(open(save_path, 'rb'))
    return loaded_model

def save_dataframe(dataframe, root_dir, models_dir, model_name, model_id):
    save_path = check_saving_path(root_dir, models_dir,model_name, model_id)
    dataframe.to_csv(save_path)
    