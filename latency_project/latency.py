#!/usr/bin/env python
# coding: utf-8

# In[1]:


import yaml
import numpy as np
import pandas as pd
import timeit




class Latency:
    pass 
    
    def PrepareGroundTruth(path_GT): # for ground truth we need to keep only the rows that have both video
                                     # frame name and values (some of the frames did not have values)
        stream = open(path_GT, "r") # unpacking the data
        docs = yaml.load_all(stream)
        lis_test = []
        
        
        for doc in docs:
            lis_test.append(doc.items()) # creating the new list 
            
        new_list_test=  []
        for i in range(len(lis_test)):
            if len(list(lis_test[i])) == 2: #selecting only the rows that have both frame names and values
                new_list_test.append(lis_test[i])
        vid_frames = []
        
        for i in range(len(list(new_list_test))):
            y = list(new_list_test[i])[0][1].lstrip().lstrip().split('VD')[1].split(':') #appending frames
            vid_frames.append(y)
        test_predictions = []
        
        for i in range(len(list(new_list_test))):
            y = list(new_list_test[i])[1][1]
            test_predictions.append(y)#getting predictions

        test_prob=[]
        
        for i in range(len(test_predictions)):
            test_prob.append(test_predictions[i][0][0])#separating probabilities
        
        test_dimensions=[]
        
        for i in range(len(test_predictions)):
            test_dimensions.append(test_predictions[i][0][1])# getting dimensions
        
        # now we convert the whole results to pandas dataframes so that consequent operations become more comfortable
        
        test_df = pd.DataFrame(vid_frames, columns= { 'video_number','frame_number', })
        
        test_df['probabilities'] = pd.DataFrame(test_prob)
        
        
        df_test= pd.DataFrame(test_dimensions, columns = {'x_min',  'y_min', 'x_max', 'y_max'})
        
        df_test.columns = 'x_min',  'y_min', 'x_max', 'y_max'
        
        test_data =  pd.merge(test_df, df_test, left_index = True, right_index=True)
        
        test_data.columns = ['video_number', 'frame_number', 'probabilities', 'x_min',  'y_min',  'x_max', 'y_max']
        # we calculate the area of the ground truth
        test_data['area'] = (test_data['x_max']-  test_data['x_min'] + 1) *(test_data['y_max']-  test_data['y_min'] + 1)
        
        return test_data
        
            
    # here are perform basically the same operations but we also have to keep the frames that do not have corresponding predictions and dimensions because in doing so we will be able to preciesly answer how many frames have passed since object has been seen in ground truth
    
    def PreparePredictionsData(path_predictions):
        stream = open(path_predictions, "r")
        docs = yaml.load_all(stream)
        lis_pred = []
        for doc in docs:
            lis_pred.append(doc.items())
    
        pred_frames = []
        for i in range(len(list(lis_pred))):
            y = list(lis_pred[i])[0][1].lstrip().lstrip().split('VD')[1].split(':')
            pred_frames.append(y)
        
        
        pred_predictions = []
        for i in range(len(list(lis_pred))):
            if len(lis_pred[i]) == 1:
                pred_predictions.append([[0,[0,0,0,0]]])# if there is no pred/dimensions we will add 0-s
            else:
                y = list(lis_pred[i])[1][1]
                pred_predictions.append(y)
        
        prediction_dimensions=[]
        for i in range(len(pred_predictions)):
            prediction_dimensions.append(pred_predictions[i][0][1])
            
            
        prediction_prob=[]
        for i in range(len(pred_predictions)):
            prediction_prob.append(pred_predictions[i][0][0])
            
            # converting to pandas
        pred_df = pd.DataFrame(pred_frames, columns= { 'frame_number','video_number'})
        
        pred_df['probabilities'] = pd.DataFrame(prediction_prob)
        
        df= pd.DataFrame(prediction_dimensions, columns = {'x_min',  'y_min', 'x_max', 'y_max'})
        
        df.columns = 'x_min',  'y_min', 'x_max', 'y_max'
        
        pred_data =  pd.merge(pred_df, df, left_index = True, right_index=True)
        
        
        pred_data.columns = ['video_number', 'frame_number', 'probabilities', 'x_min', 'y_min',  'x_max', 'y_max']
        
        pred_data['area'] = (pred_data['x_max']-  pred_data['x_min'] + 1) *(pred_data['y_max']-  pred_data['y_min'] + 1)
        
        
        return pred_data
        
        
     # we calculate the latency, first, we will search for every frame in ground truth how many frame has been passed since we meet the same frame in predictions and then we will see if this frame meets our conditions, if it does not, we  update the frame by one and calculate corresponding latency value
    def LatencyCalculation(test_data, pred_data):
        latency = []
        for i in range(len(test_data['frame_number'])):
            for j in range(4000):#There are around 6602 frames with values in GT and overall 10000 overall 
                                 # we have to search in every frame and taking high value of 6602 + 4000                                    will allow us to do so, 
                                   # try catch are help us not to break loop of index exceeds the value
                try:
                    if  (test_data['frame_number'][i] == pred_data['frame_number'][i+j]) & (test_data['video_number'][i] == pred_data['video_number'][i+j]): #finding the corresponding frame from GT in predictions
                        for lat in range(250):#from my experiment I did not find any latency of higher than 250 hence we will add at most 250 to see if there is a match # check if conditions are satisfied
                            if (test_data['video_number'][i] == pred_data['video_number'][i+j+lat]) & (pred_data['probabilities'] [i+j+lat]> 0.25) & (((max(test_data['x_max'][i], pred_data['x_max'][i+j+lat]) - max(test_data['x_min'][i], pred_data['x_min'][i+j+lat]) +1) * (max(test_data['y_max'][i], pred_data['y_max'][i+j+lat]) - max(test_data['y_min'][i], pred_data['y_min'][i+j+lat]) +1)) / (float(test_data['area'][i] +pred_data['area'][i+j+lat]) - (max(test_data['x_max'][i], pred_data['x_max'][i+j+lat]) - max(test_data['x_min'][i], pred_data['x_min'][i+j+lat]) +1) * (max(test_data['y_max'][i], pred_data['y_max'][i+j+lat]) - max(test_data['y_min'][i], pred_data['y_min'][i+j+lat]) +1)) > 0.01):
                                latency.append([['vid_number', test_data['video_number'][i]],['frame',test_data['frame_number'][i]], ['latency', lat]])
                                break # if there is a match we break the loop

                        break
                except:
                    pass
        
        
        vid_number =[]
        for i in range(len(latency)):
            vid_number.append(latency[i][0][1])
        
        
        frame_number =[]
        for i in range(len(latency)):
            frame_number.append(latency[i][1][1])
            
        lat =[]
        for i in range(len(latency)):
            lat.append(latency[i][2][1])
        
        # we do some additional operations to recieve values in most user friendly way
        vid_number=pd.DataFrame(vid_number, columns = {'vid_number'})
        frame_number = pd.DataFrame(frame_number, columns = {'frame_number'})
        lat = pd.DataFrame(lat, columns = {'lat'})
        final =  pd.merge(vid_number,  frame_number, left_index= True, right_index =True)
        end =  pd.merge(final,  lat, left_index= True, right_index =True)
        
        end['full_video_name'] =  np.where(end['vid_number'] == '13', 'AsuMayoTest_clean:testVD13', 
         (np.where(end['vid_number'] == '6', 'AsuMayoTest_clean:testVD6',
         (np.where(end['vid_number'] == '8', 'AsuMayoTest_clean:testVD8',
        (np.where(end['vid_number'] == '4', 'AsuMayoTrain_clean:ShortVD4', 
        (np.where(end['vid_number'] == '68', 'AsuMayoTrain_clean:ShortVD68', 
         (np.where(end['vid_number'] == '69', 'AsuMayoTrain_clean:ShortVD69', 


         (np.where(end['vid_number'] == '010085', 'ELI:ShortVD010085', 
         (np.where(end['vid_number'] == '010086', 'ELI:ShortVD010086',
                 (np.where(end['vid_number'] == '010086', 'ELI:ShortVD010086',
                (np.where(end['vid_number'] == '010090', 'ELI:ShortVD010090',
                (np.where(end['vid_number'] == '010096', 'ELI:ShortVD010096',
                (np.where(end['vid_number'] == '020084', 'ELI:ShortVD020084',
                (np.where(end['vid_number'] == '030090', 'ELI:ShortVD030090',
                (np.where(end['vid_number'] == '040090', 'ELI:ShortVD040090',
                (np.where(end['vid_number'] == '0050085', 'ELI:ShortVD050085','ELI:ShortVD050085'
                         )))))))))))))))))))))))))))))
        
        return end
    
    # getting average per video and max per video
    def getAverage(latency):
        agg = latency.groupby(['full_video_name']).agg({'lat': 'mean'}).reset_index()
        return agg
    
    
    def getMax(latency):
        agg = latency.groupby(['full_video_name']).agg({'lat': 'max'}).reset_index()
        return agg
        
        
    
    





          

     


            
            
            
            
        
            
        
                
    

        
 
        
        
        
    
    
    
    

    
    
    
    
    
    

    



            
            
    
    
    

    






