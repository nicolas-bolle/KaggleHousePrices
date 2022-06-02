# For all the ugly code, like converting things to ordinals

import pandas as pd
import numpy as np

# Goes through and converts every column that should be ordinal; works in place
def ordinals(df):
    
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
    
    # PoolQC
    convert = convert_main_NA
    df['PoolQC'] = [nan(convert,s) for s in df['PoolQC']]
    
    # Fence
    convert = {'NA' : 0, 'MnWw' : 1, 'GdWo' : 2, 'MnPrv' : 3, 'GdPrv' : 4}
    df['Fence'] = [nan(convert,s) for s in df['Fence']]
    
    # MiscFeature
    convert = {'NA' : 0, 'Elev' : 1, 'Gar2' : 1, 'Shed' : 1, 'TenC' : 1}
    df['MiscFeature'] = [nan(convert,s) for s in df['MiscFeature']]
    
# Impute variables
def impute(df):
    # LotFrontage
    return

# Processes variables with standard transformations and scalings
def process(df):
    ## SalePrice
    if 'SalePrice' in df.columns:
        df['SalePrice'] = np.log(df['SalePrice'])
        
    ## Street
    df.drop(columns='Street', inplace=True)
    
    ## Utilities
    df.drop(columns='Utilities', inplace=True)
    
    ## Lot
    df['LotArea'] = np.log(df['LotArea'])
    df['LotFrontage'] = np.log(df['LotFrontage'])
    
    ## Porch
    # Porch square footage, log1p scaled
    df['PorchSQ'] = np.log(df['OpenPorchSF'] + df['EnclosedPorch'] + df['3SsnPorch'] + df['ScreenPorch'] + 1)
    # Ordinal: 0 = no porch, 1 = basic porch, 2 = enclosed porch
    df['Porch'] = 0
    df.loc[df['PorchSQ'] > 0,'Porch'] = 1
    df.loc[df['EnclosedPorch'] + df['3SsnPorch'] + df['ScreenPorch'] > 0,'Porch'] = 2
    # Delete old variables
    df.drop(columns=['OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch'], inplace=True)
    
    ## Pool and MiscFeature
    df.loc[df['PoolQC'] > 0, 'MiscFeature'] = 1
    df.drop(columns=['PoolArea','PoolQC'], inplace=True)
    
    ## Garage
    df.drop(columns=['GarageArea','GarageCond'], inplace=True)
    
    ## Basement
    # Rather than have 2 sets of basement area variables, I'll just keep the one with the better quality
    # Loses some detail, but should capture the basic idea
    df['BsmtFinType'] = df[['BsmtFinType1','BsmtFinType2']].max(axis=1)
    pick = df['BsmtFinType1'] > df['BsmtFinType2']
    df.loc[pick, 'BsmtFinSF'] = df.loc[pick, 'BsmtFinSF1']
    df.loc[~pick, 'BsmtFinSF'] = df.loc[~pick, 'BsmtFinSF2']
    
    ## Bathrooms
    df['FullBath'] = df['FullBath'] + df['BsmtFullBath']
    df['HalfBath'] = df['HalfBath'] + df['BsmtHalfBath']
    df.drop(columns=['BsmtFullBath','BsmtHalfBath'], inplace=True)
    
    