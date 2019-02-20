
class Visualize):
    def __init__(self,df,model):
        self.df = df
        self.model = model
        self.figsize = (15,15)
        self.prediction_threshold = None
        self.index = None
        self.mode = None
        self.img = None
        self.msk = None
        self.prediction = None
        
    def show(self,index,ax=None,mode='image'):
        self.mode = mode
        
        if index == 'random':
            index = np.random.random_integers(0,len(self.df))
            #self.index = self.df[self.df['Images'].str.contains(str(index))].index[0]
            self.index = index
        else:
            self.index = index
            
        self.__load_data()
        
        if self.mode == "image":
            if ax == None:
                fig, ax = plt.subplots(figsize=self.figsize)
            ax.imshow(self.img)
            ax.set_title('Image Nr: ' + str(self.df['Images'][self.index]))
        if self.mode == "prediction":
            if ax == None:
                fig, ax = plt.subplots(figsize=self.figsize)
            ax.imshow(self.msk)
            ax.set_title('Image Nr: ' + str(self.df['Images'][self.index]))
        if self.mode == "image_prediction":
            self.__predict()
            if ax == None:
                fig, ax = plt.subplots(figsize=self.figsize)
            ax.imshow(self.img)
            ax.imshow(self.prediction>self.prediction_threshold, alpha=0.5)
            ax.set_title('Image Nr: ' + str(self.df['Images'][self.index]))
        if self.mode == "image_prediction_error":
            self.__predict()
            if ax == None:
                fig, ax = plt.subplots(figsize=self.figsize)
                
            #prediction_f = np.array((self.prediction>self.prediction_threshold).flatten)
            #msk_f = np.array((self.msk<1).flatten)
            #print(msk_f.shape)
            #intersection = np.multiply(prediction_f,msk_f)
            
            error = np.equal(self.prediction>0.95,self.msk<1)
            ax.imshow(self.img)
            if self.prediction_threshold == None:
                ax.imshow(self.prediction, alpha=0.4)
            else:
                ax.imshow(self.prediction>self.prediction_threshold, alpha=0.4)
            ax.imshow(error,cmap='Reds', alpha=0.3)
            overlap = np.invert(error)
            dice = ((overlap.sum()/(error.shape[0]*error.shape[1])))
            dice = overlap.sum()/(self.msk.sum())
            if dice > 1:
                dice = 1/dice
            ax.set_title('Image : ' + str(self.df['Images'][self.index]) + "    Dice Coeff.: " + str(round(dice, 2)))
                         
    def show_matrix(self,index,mode='image'):
        self.mode = mode
        if index == 'random':
            index = []
            n = 8
            for i in range(n):
                tmp_index = np.random.random_integers(0,len(self.df))
                index.append(tmp_index)
        else:
            n = len(index)
        #Add None if odd:
        if n%2 != 0:
            index.append(None)
                         
        fig, ax = plt.subplots(int(n/2),2,figsize=(20,4*n))
        
        for i in log_progress(range(len(ax)),every=1):
            ind = index[i:i+2]
            self.__make_image_row(ax[i],ind)   
                         
    def __make_image_row(self,ax,index):
        self.show(index[0],ax[0],self.mode)
        self.show(index[1],ax[1],self.mode)
        
    def __load_data(self):
        img= imread(DATA_IMAGE_PATH+"/"+self.df["Images"].values[self.index])
        msk = imread(DATA_MASK_PATH+"/"+self.df["Masks"].values[self.index])
        self.img = resize(img,(SHAPE[0],SHAPE[1])).reshape(*SHAPE,3)
        self.msk = resize(msk,(SHAPE[0],SHAPE[1])).reshape(*SHAPE)
        
    def __predict(self):
        #tmp_img = np.zeros((n,*self.img.shape))
        tmp_img = self.img.reshape(1,*SHAPE,3)
        self.prediction = model.predict(tmp_img)
        self.prediction = self.prediction.reshape(*SHAPE)