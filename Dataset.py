import numpy as np

class Dataset:
    '''Synthetic hiring data with numpy'''

    def __init__(self):
        pass
    
    def generate(size = 5000):
        
        # reproducibility
        np.random.seed(10)

        # feature setup
        degree = np.random.choice(a=range(4), size=size)
        age = np.random.choice(a=range(18,61), size=size)
        gender = np.random.choice(a=range(2), size=size)
        major = np.random.choice(a=range(8), size=size)
        gpa = np.round(np.random.normal(loc=2.90, scale=0.5, size=size), 2)
        experience = np.empty(size)  
        bootcamp = np.random.choice(a=range(2), size=size)
        github = np.random.choice(a=range(21), size=size)
        blogger = np.random.choice(a=range(2), size=size)
        articles = np.empty(size) 
        t1 = np.empty(size) 
        t2 = np.empty(size) 
        t3 = np.empty(size) 
        t4 = np.empty(size) 
        t5 = np.empty(size) 
        hired = np.empty(size) 

        # Arrange features into ndarray
        data = np.vstack((degree, age, gender, major, gpa, experience, bootcamp, github, blogger, articles,hired,t1,t2,t3,t4,t5)).T

        # Constrain GPA 
        data[:,4][data[:,4] > 4]  = 4
        data[:,4][data[:,4] < 1]  = 1

        # # Set experience based on age
        experience = lambda t: np.random.choice(a=range(0, int(t)-17))
        v_experience = np.vectorize(experience)
        data[:, 5] = v_experience(data[:, 1])

        # Set number of articles if blogger flag
        num_articles = lambda x : np.random.choice(a=range(1,21),size = 1) if x == 1 else 0
        v_num_articles = np.vectorize(num_articles)
        data[:, 9] = v_num_articles(data[:, 8])

        # Set target flags

        # Degree
        def degree_target_fun(x):
            if x == 0:
                return np.random.choice(a=range(2), size=1, p=[0.92, 0.08]) ## no bachelors
            elif x == 1:
                return np.random.choice(a=range(2), size=1, p=[0.30, 0.70]) ## bachelors
            elif x == 2:
                return np.random.choice(a=range(2), size=1, p=[0.20, 0.80]) ## masters
            else:
                return np.random.choice(a=range(2), size=1, p=[0.80, 0.20]) ## PhD

          
        degree_target_fun_v = np.vectorize(degree_target_fun)
        data[:,11] = degree_target_fun_v(data[:, 0])

        # Experience
        def experience_target_fun(x):
            if x <= 10:
                return np.random.choice(a=range(2), size=1, p=[0.10, 0.90]) ## <= 10 yrs exp
            elif x <= 25:
                return np.random.choice(a=range(2), size=1, p=[0.80, 0.20]) ## 11-25 yrs exp
            else:
                return np.random.choice(a=range(2), size=1, p=[0.95, 0.05]) ## >= 26 yrs exp

        experience_target_fun_v = np.vectorize(experience_target_fun)
        data[:,12] = experience_target_fun_v(data[:, 5])

        # Bootcamp
        def bootcamp_target_fun(x):
            if x == 1:
                return np.random.choice(a=range(2), size=1, p=[0.25, 0.75]) ## bootcamp
            else:
                return np.random.choice(a=range(2), size=1, p=[0.50, 0.50]) ## no bootcamp

        bootcamp_target_fun_v = np.vectorize(bootcamp_target_fun)
        data[:,13] = bootcamp_target_fun_v(data[:, 6])

        # Github
        def github_target_fun(x):
            if x == 0:
                return np.random.choice(a=range(2), size=1, p=[0.95, 0.05]) ## 0 projects
            elif x <= 5:
                return np.random.choice(a=range(2), size=1, p=[0.35, 0.65]) ## 1-5 projects
            else:
                return np.random.choice(a=range(2), size=1, p=[0.05, 0.95]) ## > 5 projects

        github_target_fun_v = np.vectorize(github_target_fun)
        data[:,14] = github_target_fun_v(data[:, 7])

        # Blogger
        def blogger_target_fun(x):
            if x == 1:
                return np.random.choice(a=range(2), size=1, p=[0.30, 0.70]) ## blogger
            else:
                return np.random.choice(a=range(2), size=1, p=[0.50, 0.50]) ## !blogger

        blogger_target_fun_v = np.vectorize(blogger_target_fun)
        data[:,15] = blogger_target_fun_v(data[:, 8])

        #Sum targets
        tgt = data[:,11] + data[:,12] + data[:,13] + data[:,14] + data[:,15]

        # Convert to binary feature (hired yes or no)
        tgt[tgt>=3] = 1
        tgt[tgt>1] = 0

        # Set Hired value
        data[:,10] = tgt

        # Remove target flags
        np.delete(data, [11,15], axis=1)

        # flip some of the predictions (adds complexity to modeling)
        np.random.seed(15)
        percent_to_flip = 0.03  ## % of hired values to flip
        num_to_flip = int(np.floor(percent_to_flip * len(data)))  ## determine number of hired values to flip

        #create mask, choose random indices from x according to pdf, set chosen indices to True:
        indices = np.full(data.shape[0], False, bool)

        randices = np.random.choice(np.arange(indices.shape[0]), num_to_flip, replace=False)  

        indices[randices] = True

        x_rand_vals = data[randices]
        x_remaining = data[~indices]

        z = np.copy(x_rand_vals)

        x_rand_vals[:,10][z[:,10]==1] = 0
        x_rand_vals[:,10][z[:,10]==0] = 1

        # Join the flipped data back to x_remaining
        data = np.vstack((x_rand_vals,x_remaining))

        # Ranomly shuffle array in place
        np.random.shuffle(data)

        return data

import pandas as pd

class Dataset_Original:
    '''Dataset code as provided by David Ziganto'''

    def __init__(self):
        pass
    
    def generate(size = 5000):

        # reproducibility
        np.random.seed(10)

        degree = np.random.choice(a=range(4), size=size)
        age = np.random.choice(a=range(18,61), size=size)
        gender = np.random.choice(a=range(2), size=size)
        major = np.random.choice(a=range(8), size=size)
        gpa = np.round(np.random.normal(loc=2.90, scale=0.5, size=size), 2)
        experience = None  
        bootcamp = np.random.choice(a=range(2), size=size)
        github = np.random.choice(a=range(21), size=size)
        blogger = np.random.choice(a=range(2), size=size)
        articles = 0  
        t1, t2, t3, t4, t5 = None, None, None, None, None
        hired = 0

        mydict = {"degree":degree, "age":age, 
                "gender":gender, "major":major, 
                "gpa":gpa, "experience":experience, 
                "github":github, "bootcamp":bootcamp, 
                "blogger":blogger, "articles":articles,
                "t1":t1, "t2":t2, "t3":t3, "t4":t4, "t5":t5, "hired":hired}

        df = pd.DataFrame(mydict, 
                                columns=["degree", "age", "gender", "major", "gpa", 
                                            "experience", "bootcamp", "github", "blogger", "articles",
                                            "t1", "t2", "t3", "t4", "t5", "hired"])

        np.random.seed(42)

        for i, _ in df.iterrows(): 
            
            # Constrain GPA
            if df.loc[i, 'gpa'] < 1.00 or df.loc[i, 'gpa'] > 4.00:
                if df.loc[i, 'gpa'] < 1.00:
                    df.loc[i, 'gpa'] = 1.00
                else:
                    df.loc[i, 'gpa'] = 4.00
            
            # Set experience based on age
            df.loc[i, 'experience'] = np.random.choice(a=range(0, df.loc[i, 'age']-17))    
            
            # Set number of articles if blogger flag
            if df.loc[i, 'blogger']:
                df.loc[i, 'articles'] = np.random.choice(a=range(1, 21), size=1) 
            
            # Set target flags
            for feature in ['degree', 'experience', 'bootcamp', 'github', 'blogger']:
                if feature == 'degree':  
                    if df.loc[i, feature] == 0:
                        df.loc[i, 't1'] = int(np.random.choice(a=range(2), size=1, p=[0.92, 0.08])) ## no bachelors
                    elif df.loc[i, feature] == 1:
                        df.loc[i, 't1'] = int(np.random.choice(a=range(2), size=1, p=[0.30, 0.70])) ## bachelors
                    elif df.loc[i, feature] == 2:
                        df.loc[i, 't1'] = int(np.random.choice(a=range(2), size=1, p=[0.20, 0.80])) ## masters
                    else:
                        df.loc[i, 't1'] = int(np.random.choice(a=range(2), size=1, p=[0.80, 0.20])) ## PhD
                elif feature == 'experience':
                    if df.loc[i, feature] <= 10:
                        df.loc[i, 't2'] = int(np.random.choice(a=range(2), size=1, p=[0.10, 0.90])) ## <= 10 yrs exp
                    elif df.loc[i, feature] <= 25:
                        df.loc[i, 't2'] = int(np.random.choice(a=range(2), size=1, p=[0.80, 0.20])) ## 11-25 yrs exp
                    else:
                        df.loc[i, 't2'] = int(np.random.choice(a=range(2), size=1, p=[0.95, 0.05])) ## >= 26 yrs exp
                elif feature == 'bootcamp':
                    if df.loc[i, feature]:
                        df.loc[i, 't3'] = int(np.random.choice(a=range(2), size=1, p=[0.25, 0.75])) ## bootcamp
                    else:
                        df.loc[i, 't3'] = int(np.random.choice(a=range(2), size=1, p=[0.50, 0.50])) ## no bootcamp
                elif feature == 'github':
                    if df.loc[i, feature] == 0:
                        df.loc[i, 't4'] = int(np.random.choice(a=range(2), size=1, p=[0.95, 0.05])) ## 0 projects
                    elif df.loc[i, feature] <= 5:
                        df.loc[i, 't4'] = int(np.random.choice(a=range(2), size=1, p=[0.35, 0.65])) ## 1-5 projects
                    else:
                        df.loc[i, 't4'] = int(np.random.choice(a=range(2), size=1, p=[0.05, 0.95])) ## > 5 projects
                else:
                    if df.loc[i, feature]:
                        df.loc[i, 't5'] = int(np.random.choice(a=range(2), size=1, p=[0.30, 0.70])) ## blogger
                    else:
                        df.loc[i, 't5'] = int(np.random.choice(a=range(2), size=1, p=[0.50, 0.50])) ## !blogger
            
            # Set hired value
            if (df.loc[i, 't1'] + df.loc[i, 't2'] + df.loc[i,'t3'] + df.loc[i,'t4'] + df.loc[i, 't5']) >= 3:
                df.loc[i, 'hired'] = 1

        # Drop target flags        
        df.drop(df[['t1', 't2', 't3', 't4', 't5']], axis=1, inplace=True)

        # Set 'experience' to numeric (was object type)
        df['experience'] = df['experience'].apply(pd.to_numeric)

        # adds complexity to modeling
        np.random.seed(15)

        percent_to_flip = 0.03  ## % of hired values to flip
        num_to_flip = int(np.floor(percent_to_flip * len(df)))  ## determine number of hired values to flip
        flip_idx = np.random.randint(low=0, high=len(df), size=num_to_flip)  ## randomly select indices

        for i, _ in df.loc[flip_idx].iterrows(): 
            if df.loc[i, 'hired'] == 1:
                df.loc[i, 'hired'] = 0
            else:
                df.loc[i, 'hired'] = 1


        return df