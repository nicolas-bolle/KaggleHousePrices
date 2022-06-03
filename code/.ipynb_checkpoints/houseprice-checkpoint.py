# All the ugly code

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from collections import Counter
from sklearn.preprocessing import StandardScaler

# A class with sklearn-like interface for doing all the data preprocessing, including inputation trained on the training data
# No .fit() method, just use fit_transform()
class Preprocess():
    
    # Undo the StandardScaler() and log1p on the SalePrice column
    def unscale(self,y):
        return np.exp(self.ScalerTarget.inverse_transform(y)) - 1
    
    # Preprocesses the training data, and fits the imputers to it along the way
    def fit_transform(self,train_data):
        
        if not 'SalePrice' in train_data.columns:
            return 'Error: no SalePrice column'
        
        # 1. Basic processing
        df = train_data.copy()
        self.ordinals(df)
        self.process(df)
        self.impute(df)
        
        # 2. Processing that fits onto the data
        
        # Variable types
        self.numeric = list(df.select_dtypes(include = ['float']).columns)
        self.numeric.remove('LotFrontage')
        self.numeric.remove('SalePrice')
        self.ordinal = list(df.select_dtypes(include = ['int']).columns)
        self.categorical = list(df.select_dtypes(include = ['object']).columns)
        
        # Fit basic imputers
        self.ImputerNumeric = SimpleImputer(strategy='median')
        self.ImputerOrdinal = SimpleImputer(strategy='most_frequent')
        
        # Apply basic imputers
        df[self.numeric] = self.ImputerNumeric.fit_transform(df[self.numeric])
        df[self.ordinal] = self.ImputerOrdinal.fit_transform(df[self.ordinal])
        
        # Fit (manual-ish) LotFrontage imputer; a linear fit in terms of LotArea
        NArows = df['LotFrontage'].isna()
        self.LFreg = LinearRegression().fit(df.loc[~NArows,'LotArea'].values.reshape(-1,1), df.loc[~NArows,'LotFrontage'].values.reshape(-1,1))
        
        # Apply the LotFrontage imputer
        df.loc[NArows,'LotFrontage'] = self.LFreg.predict(df.loc[NArows,'LotArea'].values.reshape(-1,1))
        
        # Fit a custom imputer to categorical features
        self.Imputes = []
        for col in self.categorical:
            # Tally the non NA values
            NArows = df[col].isna()
            tallies = Counter(list(df.loc[~NArows,col]))
            
            # Pick the most frequent one
            value = max(tallies.values())
            key = next(key for key, val in tallies.items() if val == value)
            self.Imputes.append(key)
            
            # Impute with it
            df.loc[NArows,col] = key
        
        # get_dummies on categoricals
        df = df.join(pd.get_dummies(df[self.categorical])).drop(columns=self.categorical)
        
        # Save the final set of feature names
        self.columns = df.columns
        
        # Fit and apply standard scalers; separate one for SalePrice, just to make inverting it easier
        self.ScalerMain   = StandardScaler()
        self.ScalerTarget = StandardScaler()
        self.main = df.columns.drop('SalePrice')
        df[self.main] = self.ScalerMain.fit_transform(df[self.main].values)
        df['SalePrice'] = self.ScalerTarget.fit_transform(df['SalePrice'].values.reshape(-1,1))
        
        return df
    
    
    # Preprocesses the test data, using the previously fit things
    def transform(self,train_data):    
        
        # 1. Basic processing
        df = train_data.copy()
        
        # For simplicity, fill in SalePrice as NA if it doesn't exist
        if not 'SalePrice' in df.columns:
            df['SalePrice'] = np.nan
        
        self.ordinals(df)
        self.process(df)
        self.impute(df)
        
        # 2. Using the previously fit things
        
        # Apply basic imputers
        df[self.numeric] = self.ImputerNumeric.transform(df[self.numeric])
        df[self.ordinal] = self.ImputerOrdinal.transform(df[self.ordinal])
        
        # Apply the LotFrontage imputer
        NArows = df['LotFrontage'].isna()
        df.loc[NArows,'LotFrontage'] = self.LFreg.predict(df.loc[NArows,'LotArea'].values.reshape(-1,1))
        
        # Apply the custom imputer to categorical features
        for i in range(len(self.categorical)):
            col = self.categorical[i]
            df.loc[df[col].isna(),col] = self.Imputes[i]
        
        # get_dummies on categoricals
        df = df.join(pd.get_dummies(df[self.categorical]))
        
        # If any columns are missing, set them to zeros; but we'll take SalePrice to be NA
        for col in self.columns:
            if not col in df.columns:
                df[col] = 0
        
        # Pick out the columns as we had them when training at this step
        df = df[self.columns]
        
        # Apply standard scalers
        df[self.main] = self.ScalerMain.transform(df[self.main].values)
        df['SalePrice'] = self.ScalerTarget.transform(df['SalePrice'].values.reshape(-1,1))
        
        return df

    
    # Goes through and converts every column that should be ordinal; works in place
    def ordinals(self,df):

        # Wrapper to deal with nan values and things not in the dictionary
        def nan(convert, s):
            # Easy case
            if s in convert:
                return convert[s]
            # If we have NA and that's "in" the dictionary
            elif pd.isna(s) and 'NA' in convert:
                return convert['NA']
            # Otherwise, just return NA
            else:
                return np.nan

        # Standard converters that we'll use a few times
        convert_main_NA = {'NA' : 0, 'Po' : 1, 'Fa' : 2, 'TA' : 3, 'Gd' : 4, 'Ex' : 5}
        convert_main    = {'Po' : 1, 'Fa' : 2, 'TA' : 3, 'Gd' : 4, 'Ex' : 5}

        # Street
        convert = {'Grvl' : 1, 'Pave' : 2}
        df['Street'] = [nan(convert,s) for s in df['Street']]

        # Alley
        convert = {'NA' : 0, 'Grvl' : 1, 'Pave' : 1} # Making it yes/no for alley access
        df['Alley'] = [nan(convert,s) for s in df['Alley']]

        # LotShape
        convert = {'IR3' : 1, 'IR2' : 2, 'IR1' : 3, 'Reg' : 4}
        df['LotShape'] = [nan(convert,s) for s in df['LotShape']]

        # Utilities
        convert = {'ELO' : 1, 'NoSeWa' : 2, 'NoSewr' : 3, 'AllPub' : 4}
        df['Utilities'] = [nan(convert,s) for s in df['Utilities']]

        # LandSlope
        convert = {'Gtl' : 1, 'Mod' : 2, 'Sev' : 3}
        df['LandSlope'] = [nan(convert,s) for s in df['LandSlope']]

        # ExterQual
        convert = convert_main
        df['ExterQual'] = [nan(convert,s) for s in df['ExterQual']]

        # ExterCond
        convert = convert_main
        df['ExterCond'] = [nan(convert,s) for s in df['ExterCond']]

        # BsmtQual
        convert = convert_main_NA
        df['BsmtQual'] = [nan(convert,s) for s in df['BsmtQual']]

        # BsmtCond
        convert = convert_main_NA
        df['BsmtCond'] = [nan(convert,s) for s in df['BsmtCond']]

        # BsmtExposure
        convert = {'NA' : 0, 'No' : 1, 'Mn' : 2, 'Av' : 3, 'Gd' : 4}
        df['BsmtExposure'] = [nan(convert,s) for s in df['BsmtExposure']]

        # BsmtFinType1
        convert = {'NA' : 0, 'Unf' : 1, 'LwQ' : 2, 'Rec' : 3, 'BLQ' : 4, 'ALQ' : 5, 'GLQ' : 6}
        df['BsmtFinType1'] = [nan(convert,s) for s in df['BsmtFinType1']]

        # BsmtFinType2
        convert = {'NA' : 0, 'Unf' : 1, 'LwQ' : 2, 'Rec' : 3, 'BLQ' : 4, 'ALQ' : 5, 'GLQ' : 6}
        df['BsmtFinType2'] = [nan(convert,s) for s in df['BsmtFinType2']]

        # HeatingQC
        convert = convert_main
        df['HeatingQC'] = [nan(convert,s) for s in df['HeatingQC']]

        # CentralAir
        convert = {'N' : 0, 'Y' : 1}
        df['CentralAir'] = [nan(convert,s) for s in df['CentralAir']]

        # Electrical
        convert = {'FuseF' : 0, 'FuseA' : 0, 'FuseP' : 0, 'Mix' : 0, 'SBrkr' : 1, 'NA' : 1}
        df['Electrical'] = [nan(convert,s) for s in df['Electrical']]

        # KitchenQual
        convert = convert_main
        df['KitchenQual'] = [nan(convert,s) for s in df['KitchenQual']]

        # Functional
        convert = {'NA' : 8, 'Sal' : 1, 'Sev' : 2, 'Maj2' : 3, 'Maj1' : 4, 'Mod' : 5, 'Min2' : 6, 'Min1' : 7, 'Typ' : 8}
        df['Functional'] = [nan(convert,s) for s in df['Functional']]

        # FireplaceQu
        convert = convert_main_NA
        df['FireplaceQu'] = [nan(convert,s) for s in df['FireplaceQu']]

        # GarageType
        convert = {'NA' : 0, 'Detchd' : 1, 'CarPort' : 1, 'BuiltIn' : 2, 'Basment' : 2, 'Attchd' : 2}
        df['GarageType'] = [nan(convert,s) for s in df['GarageType']]

        # GarageFinish
        convert = {'NA' : 0, 'Unf' : 1, 'RFn' : 2, 'Fin' : 3}
        df['GarageFinish'] = [nan(convert,s) for s in df['GarageFinish']]

        # GarageQual
        convert = convert_main_NA
        df['GarageQual'] = [nan(convert,s) for s in df['GarageQual']]

        # GarageCond
        convert = convert_main_NA
        df['GarageCond'] = [nan(convert,s) for s in df['GarageCond']]

        # PavedDrive
        convert = {'N' : 0, 'P' : 1, 'Y' : 2}
        df['PavedDrive'] = [nan(convert,s) for s in df['PavedDrive']]

        # PoolQC
        convert = convert_main_NA
        df['PoolQC'] = [nan(convert,s) for s in df['PoolQC']]

        # Fence
        convert = {'NA' : 0, 'MnWw' : 1, 'GdWo' : 2, 'MnPrv' : 3, 'GdPrv' : 4}
        df['Fence'] = [nan(convert,s) for s in df['Fence']]

        # MiscFeature
        convert = {'NA' : 0, 'Elev' : 1, 'Gar2' : 1, 'Shed' : 1, 'TenC' : 1}
        df['MiscFeature'] = [nan(convert,s) for s in df['MiscFeature']]

        
    # Processes variables with standard transformations and scalings
    def process(self,df):

        ## MSSubClass (it's numeric but should be categorical) - ignore for now, because it will get treated as ordinal which is more appropriate
        #df['MSSubClass'] = [str(n) for n in df['MSSubClass']]

        ## SalePrice log1p
        df['SalePrice'] = np.log(df['SalePrice'] + 1)

        ## Porch
        # Porch square footage, log1p scaled
        df['PorchSF'] = np.log(df['OpenPorchSF'] + df['EnclosedPorch'] + df['3SsnPorch'] + df['ScreenPorch'] + 1)
        # Ordinal: 0 = no porch, 1 = basic porch, 2 = enclosed porch
        df['Porch'] = 0
        df.loc[df['PorchSF'] > 0,'Porch'] = 1
        df.loc[df['EnclosedPorch'] + df['3SsnPorch'] + df['ScreenPorch'] > 0,'Porch'] = 2
        # Delete old variables
        df.drop(columns=['OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch'], inplace=True)

        ## Pool and MiscFeature
        df.loc[df['PoolQC'] > 0, 'MiscFeature'] = 1
        df.drop(columns=['PoolArea','PoolQC'], inplace=True)

        ## Basement
        # Rather than have 2 sets of basement area variables, I'll just keep the one with the better quality
        # Loses some detail, but should capture the basic idea
        df['BsmtFinType'] = df[['BsmtFinType1','BsmtFinType2']].max(axis=1)
        pick = df['BsmtFinType1'] > df['BsmtFinType2']
        df.loc[pick, 'BsmtFinSF'] = df.loc[pick, 'BsmtFinSF1']
        df.loc[~pick, 'BsmtFinSF'] = df.loc[~pick, 'BsmtFinSF2']
        df.drop(columns=['BsmtFinType1','BsmtFinType2','BsmtFinSF1','BsmtFinSF2'], inplace=True)

        ## Bathrooms
        df['FullBath'] = df['FullBath'] + df['BsmtFullBath']
        df['HalfBath'] = df['HalfBath'] + df['BsmtHalfBath']
        df.drop(columns=['BsmtFullBath','BsmtHalfBath'], inplace=True)

        ## Main removals
        df.drop(columns=['Condition2','Exterior2nd','Street','GarageCond','Utilities','Heating','LowQualFinSF'], inplace=True)

        ## Main log1ps
        colns = ['LotArea','MasVnrArea','BsmtFinSF','BsmtUnfSF','TotalBsmtSF','1stFlrSF',\
                 '2ndFlrSF','GrLivArea','GarageArea','WoodDeckSF','PorchSF','MiscVal']
        df[colns] = np.log(df[colns] + 1)


    # Impute variables
    def impute(self,df):    
        ## GarageYrBlt
        # Replace NA with the year the house was built
        rows = df['GarageYrBlt'].isna()
        df.loc[rows,'GarageYrBlt'] = df.loc[rows,'YearBuilt']