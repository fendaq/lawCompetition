class Predictor():
    import os
    import cnn_model
    import tensorflow as tf
    import numpy as np

    def  __init__(self):
        self.config=cnn_model.TCNNConfig()
        self.batch_size=config.batch_size
        self.model=cnn_model.CharLevelCNN(self.config)
        self.model_dir=os.path.dirname(__file__)+'/model/'
        self.accusation_model=self.model_dir+'accusation/'
        self.relevant_model=self.model_dir+'relevant_articles/'
        self.year_model=self.model_dir+'year/'
        self.session=tf.Session()

    def predict(self,content):
        accu_result=np.zeros(shape=len(content), dtype=np.int32)
        relvvant_result=np.zeros(shape=len(content), dtype=np.int32)
        year_result=np.zeros(shape=len(content), dtype=np.int32)
        feed_dict = {
            model.x: content,
            model.keep_prob: 1.0
        }

        self.session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess=self.session, save_path=self.accusation_model)
        accu_result=self.session.run(model.y_pred,feed_dict=feed_dict)

        saver.restore(sess=self.session, save_path=self.year_model)
        year_result=self.session.run(model.y_pred,feed_dict=feed_dict)

        saver.restore(sess=self.session, save_path=self.relevant_model)
        year_result=self.session.run(model.y_pred,feed_dict=feed_dict)

        result=[]
        for i in range(len(content)):
            temp={"accusation":accu_result[i]+1,"relevant_articles":relvvant_result[i]+1,"term_of_imprisonment":year_result[i]}
            result.append(temp)
        return result

