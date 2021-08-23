import sys
import pandas as pd
import numpy as np
import sqlite3
import math
import os
import matplotlib.pyplot as plt
from collections import defaultdict
from copy import deepcopy
import pickle
from scipy import stats
import random
from binascii import hexlify


# ------------------- PRELIMINARIES ------------------------
# Abbreviations:
#   pp - Partially precomputed
#   fp - Fully precomputed
#   aff - Adjustable function family. Used for joins in cryptdb
# Informal types:
#   JoinType = 1-1
#            | 1-Many
#            | Many-Many
#       # Note that many-to-one joins and one-to-many joins have the same
#       # properties for our purposes
#   Statistic = FP            # Fully precompute all joins
#             | PP            # Partially precompute all joins
#             | Stats         # Compute the uniform, bucketed, and full histogram
#                             # estimates
#             | None          # Get results from all possible join  annotations
# Our query language
#   To simplify parsing queries for the subset of SQL we support, we use a different
#   syntax for writing queries than regular SQL. A query has 3 forms:
#       1. (SELECT (file.attrib=value_name) q)          where q is a query
#       2. (JOIN (file1.attrib1=file2.attrib2) q1 q2)   where q1 and q2 are queries
#       3. (file_name)
# -----------------------------------------------------------


# ------------------- GLOBAL VARIABLES ------------------------


# Global Int to keep track of the total number of rows in the tables involved in
# a query
total_table_rows = 0


# Global variables to store aggregate min, max, and mean sizes for adjustable
# function family, fully precomputed, and partially precomputed joins
# They are dictionaries mapping the JoinType to the minimum, maximum, or
# mean join size in the database
aff_min_bandwidth = defaultdict(int)
fp_min_bandwidth = defaultdict(int)
pp_min_bandwidth = defaultdict(int)
aff_mean_bandwidth = defaultdict(int)
fp_mean_bandwidth = defaultdict(int)
pp_mean_bandwidth = defaultdict(int)
aff_max_bandwidth = defaultdict(int)
fp_max_bandwidth = defaultdict(int)
pp_max_bandwidth = defaultdict(int)


# Global variables to store aggregate ratios of min, max, and mean sizes between
# pp indexing and fp indexing types. Stored in JoinType to int dictionaries
pp_to_fp_min_bandwidth = defaultdict(int)
pp_to_fp_mean_bandwidth = defaultdict(int)
pp_to_fp_max_bandwidth = defaultdict(int)


# Global variables to store aff, fp, and pp mean volumes leaked and ratios
# of min, mean, and max volumes leaked from pp to fp and fp to aff. Stored in
# JoinType to int dictionaries.
# Note: pp joins leak two volumes (which rows have some equal row in the joined table),
# fp joins leak a number of volumes equivalent to the size of the intersection of
# the ranges of the joined attributes, and aff leaks volumes equivalent to the union
# of the ranges of the joined attributes.
aff_mean_leakages = defaultdict(int)
fp_mean_leakages = defaultdict(int)
pp_mean_leakages = defaultdict(int)
pp_to_fp_min_leakage = defaultdict(int)
pp_to_fp_mean_leakage = defaultdict(int)
pp_to_fp_max_leakage = defaultdict(int)
fp_to_aff_min_leakage = defaultdict(int)
fp_to_aff_mean_leakage = defaultdict(int)
fp_to_aff_max_leakage = defaultdict(int)


# Global variables used for graphing the bandwidth ratios of a join result sent back
# to the client relative to the unjoined tables against the percentage of the rows
# in the tables accessed to compute the join. Categories are of type JoinType
access_percentages = []
pp_bandwidth_ratios = []
fp_bandwidth_ratios = []
aff_bandwidth_ratios = []
categories = []


# ------------------- DATABASE AND QUERY PARSING -------------------


# Function to turn valid pandas column names into valid sql column names (no spaces, etc)
#   table - DataFrame : A pandas dataframe
#   Returns a dataframe with the attribute names changed to remove space, -, #, and %
#       characters
def cleanColNames(table):
    table.columns = table.columns.str.replace(' ', '')
    table.columns = table.columns.str.replace('-', '')
    table.columns = table.columns.str.replace('#', 'Number')
    table.columns = table.columns.str.replace('%', 'Percent')
    table.columns = table.columns.str.lower()
    return table.columns


# Helper function for parsing "file.attrib" Strings into the file and attribute
#   file_attrib : String - A string with the format file.attribute
#   Returns String * String, a tuple with the file and attribute as separate strings
def parseFileAttrib(file_attrib):
    file = file_attrib.split(".", 1)[0]
    attrib = file_attrib.split(".", 1)[1]
    return (file, attrib)


# Helper function for parsing select/join queries. Finds the matching end parenthesis
# to a query.
#   s : String - The remainder of the query, including the starting left paranthesis
#   withP : Bool - Whether the result should include parentheses around the subquery
#   Returns a String * Int tuple with the subquery and the ending index in the query
#       for the subquery
def findMatchingParen(s, withP=False):
    endIndex = 0
    numberLeft = 1
    result = ""
    begun = False
    for c in s:
        endIndex += 1
        if begun:
            if c == ")":
                numberLeft -= 1
            if c == "(":
                numberLeft += 1
            if numberLeft == 0:
                break
            result += c
        if not begun and c == "(":
            begun = True
    if withP:
        result = "(" + result + ")"
    return (result, endIndex)


# Helper function to truncate a street address into just a block. Used for creating a
# common format for joining Chicago tables where the address is specified with
# tables where only a block is given.
#   row : Series - A pandas row containing an address 
#   addr: String - The attribute name for the address column
#   Returns a String with the block
def getBlock(row, addr):
    s = row[addr]
    s_array = s.split()
    addrNumber = list(s_array[0])
    length = len(addrNumber)
    if length > 1:
        addrNumber[length - 1] = 'X'
        addrNumber[length - 2] = 'X'
    else:
        addrNumber[0] = 'X'
        addrNumber.append('X')
    # print(addrNumber)
    s_array[0] = ''.join(addrNumber)
    s_array[0] = s_array[0].rjust(5, '0')
    for i in range(len(s_array)):
        s_array[i] = s_array[i].upper()
    block = ' '.join(s_array)
    return block


# ------------------- SINGLE ATTRIBUTE JOIN EXPERIMENTS -------------


# Helper function to update the global variables for the single attribute joins experiment.
# Updates mins and maximums given info about the new single attribute join and adds
# to an accumulated sum in the means dictionary which will be divided by the number of
# joins measured in the experiments at the end
#   join_type : JoinType - The type of join
#   [aff/fp/pp]_bandwidth : Int - The bandwidth with different indexes
#   [aff/fp/pp]_leakage : Int - The number of volumes leaked with different indexes
def setGlobals(join_type, aff_bandwidth, fp_bandwidth, pp_bandwidth, aff_leakage, fp_leakage, pp_leakage):
    pp_to_fp_bandwidth = fp_bandwidth / pp_bandwidth
    pp_to_fp_leakage = fp_leakage / pp_leakage
    fp_to_aff_leakage = aff_leakage / fp_leakage
    
    if pp_to_fp_min_bandwidth[join_type] == 0:
        pp_to_fp_min_bandwidth[join_type] = pp_to_fp_bandwidth
    else:
        pp_to_fp_min_bandwidth[join_type] = min(pp_to_fp_bandwidth, pp_to_fp_min_bandwidth[join_type])
    if pp_to_fp_min_leakage[join_type] == 0:
        pp_to_fp_min_leakage[join_type] = pp_to_fp_leakage
    else:
        pp_to_fp_min_leakage[join_type] = min(pp_to_fp_leakage, pp_to_fp_min_leakage[join_type])
    if fp_to_aff_min_leakage[join_type] == 0:
        fp_to_aff_min_leakage[join_type] = fp_to_aff_leakage
    else:
        fp_to_aff_min_leakage[join_type] = min(fp_to_aff_leakage, fp_to_aff_min_leakage[join_type])
    pp_to_fp_max_bandwidth[join_type] = max(pp_to_fp_bandwidth, pp_to_fp_max_bandwidth[join_type])
    pp_to_fp_max_leakage[join_type] = max(pp_to_fp_leakage, pp_to_fp_max_leakage[join_type])
    fp_to_aff_max_leakage[join_type] = max(fp_to_aff_leakage, fp_to_aff_max_leakage[join_type])
    pp_to_fp_mean_bandwidth[join_type] += pp_to_fp_bandwidth
    pp_to_fp_mean_leakage[join_type] += pp_to_fp_leakage
    fp_to_aff_mean_leakage[join_type] += fp_to_aff_leakage
    
    if aff_min_bandwidth[join_type] == 0:
        aff_min_bandwidth[join_type] = aff_bandwidth
    else:
        aff_min_bandwidth[join_type] = min(aff_bandwidth, aff_min_bandwidth[join_type])
    if fp_min_bandwidth[join_type] == 0:
        fp_min_bandwidth[join_type] = fp_bandwidth
    else:
        fp_min_bandwidth[join_type] = min(fp_bandwidth, fp_min_bandwidth[join_type])
    if pp_min_bandwidth[join_type] == 0:
        pp_min_bandwidth[join_type] = pp_bandwidth
    else:
        pp_min_bandwidth[join_type] = min(pp_bandwidth, pp_min_bandwidth[join_type])
    aff_max_bandwidth[join_type] = max(aff_bandwidth, aff_max_bandwidth[join_type])
    fp_max_bandwidth[join_type] = max(fp_bandwidth, fp_max_bandwidth[join_type])
    pp_max_bandwidth[join_type] = max(pp_bandwidth, pp_max_bandwidth[join_type])
    aff_mean_bandwidth[join_type] += aff_bandwidth
    fp_mean_bandwidth[join_type] += fp_bandwidth
    pp_mean_bandwidth[join_type] += pp_bandwidth
    
    aff_mean_leakages[join_type] += aff_leakage
    fp_mean_leakages[join_type] += fp_leakage
    pp_mean_leakages[join_type] += pp_leakage


# Given two tables and attributes within those tables, check whether the result
# of joining those tables on those attributes will be non-empty
#   file1: String - Relation name of the first table. Only useful for printing, not needed
#       compute the result
#   table1 : Dataframe - the first table
#   attrib1 : String - The join attribute from the first table
#   file2: String - Relation name of the second table. Only useful for printing, not needed
#       compute the result
#   table2 : Dataframe - the second table
#   attrib2 : String - The join attribute from the second table
#   Returns a Bool for whether the join is non-empty
def getJoinNonEmpty(file1, table1, attrib1, file2, table2, attrib2):
    t1_dict = defaultdict(int)
    
    # Creates a dictionary mapping values to the count of occurences of rows with
    # that value at that attribute
    for val in table1[attrib1]:
        if not (val == None or (type(val)==str and val == '') or (type(val)==float and val==np.nan)):
            t1_dict[val] += 1
    t1_values = [k for k, _ in t1_dict.items()]
    
    # Checks if any value in table 2 also occurs in the hashset of table1 values
    for val in table2[attrib2]:
        if val in t1_values:
            print(file1 + " - " + attrib1 + " join " + file2 + " - " + attrib2)
            print(val)
            return True
    return False


# Takes two tables and attributes, computes the AFF, FP, and PP join stats, and
# returns the join type (1-1, 1-many, or many-many, each with a full or partial
# variety -  full meaning that >95% of the rows from both tables are included in the join,
# and partial otherwise).
#   file1: String - Relation name of the first table. Only useful for printing, not needed
#       compute the result
#   table1 : Dataframe - the first table
#   attrib1 : String - The join attribute from the first table
#   file2: String - Relation name of the second table. Only useful for printing, not needed
#       compute the result
#   table2 : Dataframe - the second table
#   attrib2 : String - The join attribute from the second table
#   connection - Defunct. No longer used, but it was the connection to the SQL instance running
#       when we were using a SQL database separately from just reading the files into pandas
#   Returns a String for whether the join is 1-1 Full/Partial, 1-Many Full/Partial,
#       or Many-Many Full/Partial (Or Empty)
def getJoinType(file1, table1, attrib1, file2, table2, attrib2, connection=None):
    t1_count = table1.shape[0]
    t2_count = table2.shape[0]
    join_count = 0
    t1_access_count = 0
    t2_access_count = 0
    
    
    # # Defunct. Used the pandas functions to interact with a SQL instance and write
    # # regular SQL queries. But this ran slower and didn't provide the granularity
    # # of information we wanted for some statistics.
    # -----------------------------------------------------------------------
    # qry = "SELECT * FROM " + file1 + " INNER JOIN " + file2 + " ON " + file1
    # qry += "." + attrib1 + "=" + file2 + "." + attrib2 + ";"
    # qry1 = "SELECT * FROM " + file1 + " WHERE EXISTS (SELECT * FROM " + file2
    # qry1 += " WHERE " + file1 + "." + attrib1 + "=" + file2 + "." + attrib2 + ");"
    # qry2 = "SELECT * FROM " + file2 + " WHERE EXISTS (SELECT * FROM " + file1
    # qry2 += " WHERE " + file1 + "." + attrib1 + "=" + file2 + "." + attrib2 + ");"
    # 
    # after_join = pd.read_sql_query(qry, connection)
    # join_count = after_join.shape[0]
    # if join_count == 0:
    #     # if not (t1_access_count == 0 and t2_access_count == 0):
    #     #     print("Something's off. Join empty but some rows accessed")
    #     return "Empty"
    # 
    # r1_table = pd.read_sql_query(qry1, connection)
    # r2_table = pd.read_sql_query(qry2, connection)
    # 
    # t1_access_count = r1_table.shape[0]
    # t2_access_count = r2_table.shape[0]
    # -----------------------------------------------------------------------
    
    # Creates dictionaries mapping values to the number of rows with that value
    t1_dict = defaultdict(int)
    t2_dict = defaultdict(int)
    for val in table1[attrib1]:
        if not (val == None or (type(val)==str and val == '') or (type(val)==float and val==np.nan)):
            t1_dict[val] += 1
    for val in table2[attrib2]:
        if not (val == None or (type(val)==str and val == '') or (type(val)==float and val==np.nan)):
            t2_dict[val] += 1
    t1_values = [k for k, _ in t1_dict.items()]
    t2_values = [k for k, _ in t2_dict.items()]
    fp_paired_volume_count = 0
    
    # Computes the number of volumes accessed, how many rows were accessed and the size
    # of the joined relation
    aff_paired_volume_count = len(t1_values) + len(t2_values)
    for t1_value in t1_values:
        if t1_value in t2_values:
            fp_paired_volume_count += 1
            join_count += t1_dict[t1_value] * t2_dict[t1_value]
            t1_access_count += t1_dict[t1_value]
            t2_access_count += t2_dict[t1_value]
    aff_paired_volume_count = len(t1_values) + len(t2_values) - fp_paired_volume_count
    
    # # Prints the bandwidths and frequencies leaked for the join
    # print("AFF/FP bandwidth: " + str(2 * join_count))
    # print("PP bandwidth: " + str(t1_access_count + t2_access_count))
    # print("AFF paired frequencies leaked: " + str(aff_paired_volume_count))
    # print("FP paired frequencies leaked: " + str(min(fp_paired_volume_count, aff_paired_volume_count)))
    # print("PP paired frequencies leaked: 2")
    
    # For non-empty joins, sets the aff, fp, and pp bandwidths and leakages,
    # the percentage of rows from the two tables accessed, and the ratios of the
    # bandwidths
    if not join_count == 0:
        aff_bandwidth = (2 * join_count) / (max(t1_count, t2_count))
        fp_bandwidth = (2 * join_count) / (max(t1_count, t2_count))
        pp_bandwidth = (t1_access_count + t2_access_count) / (max(t1_count, t2_count))
        aff_leakage = aff_paired_volume_count
        fp_leakage = min(fp_paired_volume_count + 1, aff_paired_volume_count)
        pp_leakage = 2
        
        access_percentage = (t1_access_count + t2_access_count) / (t1_count + t2_count)
        access_percentages.append(access_percentage)
        pp_bandwidth_ratios.append(pp_bandwidth)
        fp_bandwidth_ratios.append(fp_bandwidth)
        aff_bandwidth_ratios.append(aff_bandwidth)
    
    # Returns the join type and alerts us to errors
    if join_count == 0:
        if not (t1_access_count == 0 and t2_access_count == 0):
            print("Something's off. Join empty but some rows accessed")
        return "Empty"
    if access_percentage > 0.95:
        if t1_access_count == t2_access_count and t1_access_count == join_count:
            setGlobals("1-1 Full", aff_bandwidth, fp_bandwidth, pp_bandwidth, aff_leakage, fp_leakage, pp_leakage)
            categories.append("1-1")
            return "1-1 Full"
        elif t1_access_count == join_count or t2_access_count == join_count:
            setGlobals("1-Many Full", aff_bandwidth, fp_bandwidth, pp_bandwidth, aff_leakage, fp_leakage, pp_leakage)
            categories.append("1-Many")
            return "1-Many Full"
        elif join_count > max(t1_access_count, t2_access_count):
            setGlobals("Many-Many Full", aff_bandwidth, fp_bandwidth, pp_bandwidth, aff_leakage, fp_leakage, pp_leakage)
            categories.append("Many-Many")
            return "Many-Many Full"
        else:
            print("Something's off. Full join but not 1-1, 1-Many, or Many-Many")
            print("-----------------")
            print(join_count)
            print(t1_count)
            print(t2_count)
            print(t1_access_count)
            print(t2_access_count)
            print(qry)
            print("-----------------")
            return "Empty"
    if access_percentage <= 0.95:
        if t1_access_count == t2_access_count and t1_access_count == join_count:
            setGlobals("1-1 Partial", aff_bandwidth, fp_bandwidth, pp_bandwidth, aff_leakage, fp_leakage, pp_leakage)
            categories.append("1-1")
            return "1-1 Partial"
        elif t1_access_count == join_count or t2_access_count == join_count:
            setGlobals("1-Many Partial", aff_bandwidth, fp_bandwidth, pp_bandwidth, aff_leakage, fp_leakage, pp_leakage)
            categories.append("1-Many")
            return "1-Many Partial"
        elif join_count > max(t1_access_count, t2_access_count):
            setGlobals("Many-Many Partial", aff_bandwidth, fp_bandwidth, pp_bandwidth, aff_leakage, fp_leakage, pp_leakage)
            categories.append("Many-Many")
            return "Many-Many Partial" 
    print("Something's off. Partial join but not 1-1, 1-Many, or Many-Many")
    print("-----------------")
    print(join_count)
    print(t1_count)
    print(t2_count)
    print(t1_access_count)
    print(t2_access_count)
    print("-----------------")
    return "Empty"


# ------------------- COMPLEX QUERY AND HEURISTIC EXPERIMENTS -------------


# Function that will take a select/join query and output all possible PP/FP combination's
# bandwidths and bandwidth estimates
#   query : String - a query of the form
#       (SELECT (file.attrib=value_name) q)          where q is a subquery
#       (JOIN (file1.attrib1=file2.attrib2) q1 q2)   where q1 and q2 are subqueries
#       (file_name)
#   tables : Dict - a dictionary from file name to pandas table
#   values : Dict -  is a dictionary from a value name to a value. I did it this way so I
#       wouldn't need to deal with SQL types
#   statistics : Statistic - A string used to annotate the query. The variable name is
#       a bit misleading since its just general annotations, not specifically
#       annotations related to our heuristic statistics. See the Statistic
#       informal type to see the options
def processQuery(query, tables, values, statistics=None):
    print(query)
    
    # Gets the results from the query. May attempt all possible fp vs. pp
    # annotations depending on statistics
    results = getQueryTables(query, tables, values, statistics)
    
    # For each result from the possible annotations
    for res in results:
        # Extracts the tables, the connections (showing which tables have been joined 
        # and which need to be finished as post-processing), the heuristic estimates, 
        # and the results String, specifying which annotation was used (0 for PP
        # and 1 for FP)
        (resTables, resConnections, resEstimates, resString) = res
        
        # Prints the annotations
        type = ""
        for c in resString:
            if c=="0":
                type += "PP, "
            if c=="1":
                type += "FP, "
        print("JOIN types: " + type)
        
        # Sums and prints the real bandwidth and the estimated bandwidth
        bandwidth = 0
        estBandwidth = 0
        resTables = [table for _, table in resTables.items()]
        resEstimates = [est for _, est in resEstimates.items()]
        for t in resTables:
            bandwidth += t.shape[0]
        for est in resEstimates:
            estBandwidth += est
        print("Bandwidths: " + str(bandwidth))
        print("Estimated bandwidth: " + str(estBandwidth))
        print("------------------------")


# Prints the FP and PP bandwidth ratio to the total number of rows in the tables
# involved in the query
#   query : String - A valid database query
#   tables : Dict -  A dictionary mapping Strings with relation names to pandas
#       Dataframes  with the relation's tables
#   values : Dict - A dictionary mapping Strings to database values
def processBandwidth(query, tables, values):
    # Processes the query using FP indexing and extract the tables from the result
    resultsFP = getQueryTables(query, tables, values, statistics="FP")
    for res in resultsFP:
        (resTablesFP, _, _, _) = res
    
    # Processes the query using PP indexing and extract the tables from the result
    resultsPP = getQueryTables(query, tables, values, statistics="PP")
    for res in resultsPP:
        (resTablesPP, _, _, _) = res
        
    # Sum the sizes of the resulting tables for FP and PP and prints the bandwidths
    bandwidthFP = 0
    bandwidthPP = 0
    resTablesFP = [table for _, table in resTablesFP.items()]
    resTablesPP = [table for _, table in resTablesPP.items()]
    for t in resTablesFP:
        bandwidthFP += t.shape[0]
    for t in resTablesPP:
        bandwidthPP += t.shape[0]
    print("Bandwidth FP: " + str(bandwidthFP / total_table_rows))
    print("Bandwidth PP: " + str(bandwidthPP / total_table_rows))
    print("------------------------")


# Helper for processQuery (does most of the work).
#   query : String - A valid database query
#   tables : Dict -  A dictionary mapping Strings with relation names to pandas
#       Dataframes  with the relation's tables
#   values : Dict - A dictionary mapping Strings to database values
#   Returns None in the case of an error or a List where each possible distinct join annotation
#       result is a list element.
#       --------------------------------------
#       A join annotation result is of type of (Dict * List * (Dict * Dict * Dict) * String).
#       1. The first Dict contain the resulting tables, mapping String names to 
#       pandas dataframes
#       2. The list contains lists of strings called connections which specify which tables 
#       have been fully joined. All the relations named in list in connections have been
#       joined together on the server side with fp joins. If relations are in separate lists
#       in connections, then the join was partially precomputed and needs to be finished clientside.
#       For example,  if the connections list is equal to [[file1, file2, file3],
#       [file4], [file5]], then file1, file2, and file3 have been fp joined, but file4
#       and file5 have been pp joined. Therefore, selects on attributes in file1 will only
#       filter down rows in file1, file2, and file3 but not on file4 or file5.
#       3. 3-tuple of Dicts contains dictionaries for the bandwidth estimates for
#       uniform, bucketed, and full histograms respectively. For each dictionary,
#       they map relation names to the estimated number of rows from that relation's
#       table in the query using that client-side storage statistic. 
#       4. The string specifies the query's join annotation, letting us know which
#       joins have been computed as fp and which by pp. The string is binary and we use
#       0 for PP and 1 for FP
#       Note: We really should have made classes for this type of things.
def getQueryTables(query, tables, values, statistics=None):
    global total_table_rows
    
    # Finds the end of the subquery
    (q, _) = findMatchingParen(query)
    
    # Extracts the first word of the query which will either be SELECT, JOIN, or
    # a file name
    operation = q.split(" ", 1)[0]
    
    # Handles SELECT queries
    if operation == "SELECT":
        # Extracts the relation name, attribute, and selected value
        rest = q.split(" ", 1)[1]
        (value, endIndex) = findMatchingParen(rest)
        file_attrib = value.split("=", 1)[0]
        val_name = value.split("=", 1)[1]
        x = values[val_name]
        (file, attrib) = parseFileAttrib(file_attrib)
        
        # Processes the subquery
        (innerQ, _) = findMatchingParen(rest[endIndex:], withP = True)
        innerResults = getQueryTables(innerQ, tables, values, statistics)
        if innerResults == None:
            return None
        
        newResults = []
        
        # Prepares bucketed histograms if we are going to want statistics
        if statistics=="stats":
            bHist = makeBucketHist(tables, file, attrib)
        
        # For each possible result from the subquery depending on the annotations
        for result in innerResults:
            # Unwraps the tables, connections, bandwidth estimates, and annotation string
            (innerTables, connections, innerBEst, retString) = result
            # Unwraps uniform, bucketed and full from the bandwidth estimates
            (innerBEstU, innerBEstB, innerBEstF) = innerBEst
            
            # Processes the select and updates the result tables. Also updates the bandwidth
            # estimate for uniform, bucketed, and full options
            table = innerTables[file]
            # Note: Only applies the select to the relations which have been connected.
            # Cannot apply the select across partial joins
            for conn_group in connections:
                if file in conn_group:
                    accessed = list(np.where(table[attrib] == x)[0])
                    if len(accessed)==0:
                        return None
                    if statistics == "stats":
                        for conn_file in conn_group:
                            innerBEstU[conn_file] *= uniform(tables, file, attrib)
                            innerBEstB[conn_file] *= bucketHist(bHist, x)
                            innerBEstF[conn_file] *= fullHist(tables, file, attrib, x)
                            new_table = innerTables[conn_file].iloc[accessed]
                            innerTables[conn_file] = new_table
                    
                    # Updates the results for each possible annotation from the select
                    newResults.append((innerTables, connections, (innerBEstU, innerBEstB, innerBEstF), retString))
                    break
        return newResults
    
    # Handles JOIN queries
    elif operation == "JOIN":
        # Extracts the relation names and attributes
        rest = q.split(" ", 1)[1]
        (value, nextIndex) = findMatchingParen(rest)
        file_attrib1 = value.split("=", 1)[0]
        file_attrib2 = value.split("=", 1)[1]
        (file1, attrib1) = parseFileAttrib(file_attrib1)
        (file2, attrib2) = parseFileAttrib(file_attrib2)
        
        # Processes the subqueries
        (innerQ1, endIndex) = findMatchingParen(rest[nextIndex:], withP = True)
        (innerQ2, _) = findMatchingParen(rest[nextIndex + endIndex:], withP = True)
        innerResults1 = getQueryTables(innerQ1, tables, values, statistics)
        innerResults2 = getQueryTables(innerQ2, tables, values, statistics)
        newResults = []
        if innerResults1 == None or innerResults2 == None:
            return None
        
        # Makes the bucketed histograms for both attributes
        if statistics=="stats":
            bHist1 = makeBucketHist(tables, file1, attrib1)
            bHist2 = makeBucketHist(tables, file2, attrib2)
        
        # For the product of possible results from the subqueries depending on the annotations
        for result1 in innerResults1:
            for result2 in innerResults2:
                # Unpacks the results
                (innerTables1, connections1, innerBEst1, retString1) = result1
                (innerTables2, connections2, innerBEst2, retString2) = result2
                table1 = innerTables1[file1]
                table2 = innerTables2[file2]
                
                (innerBEstU1, innerBEstB1, innerBEstF1) = innerBEst1
                (innerBEstU2, innerBEstB2, innerBEstF2) = innerBEst2
                innerBEstFPU = deepcopy(innerBEstU1)
                innerBEstFPU.update(innerBEstU2)
                innerBEstFPB = deepcopy(innerBEstB1)
                innerBEstFPB.update(innerBEstB2)
                innerBEstFPF = deepcopy(innerBEstF1)
                innerBEstFPF.update(innerBEstF2)
                innerBEstPPU = deepcopy(innerBEstU1)
                innerBEstPPU.update(innerBEstU2)
                innerBEstPPB = deepcopy(innerBEstB1)
                innerBEstPPB.update(innerBEstB2)
                innerBEstPPF = deepcopy(innerBEstF1)
                innerBEstPPF.update(innerBEstF2)
                
                # Prepares new connection groups for fp and pp
                for i in range(len(connections1)):
                    conn_group = connections1[i]
                    if file1 in conn_group:
                        conn_group1 = conn_group
                        connections_temp1 = deepcopy(connections1)
                        connections_temp1.pop(i)
                        break
                for i in range(len(connections2)):
                    conn_group = connections2[i]
                    if file2 in conn_group:
                        conn_group2 = conn_group
                        connections_temp2 = deepcopy(connections2)
                        connections_temp2.pop(i)
                        break
                new_conn_group = conn_group1 + conn_group2
                
                # If we are not only computing PP joins, add FP results
                if statistics != "PP":
                    new_inner_fp = deepcopy(innerTables1)
                    new_inner_fp.update(innerTables2)
                    # Copies the table columns
                    for f in new_conn_group:
                        new_inner_fp[f] = pd.DataFrame(columns=new_inner_fp[f].columns)
                    
                    # Prepares to computes the join by generating hashtables mapping values to row
                    # indices in O(n) time and O(n) space 
                    t1_dict = defaultdict(list)
                    t2_dict = defaultdict(list)
                    for i, (_, row1) in enumerate(table1.iterrows()):
                        val = row1[attrib1]
                        if not (val == None or (type(val)==str and val == '') or (type(val)==float and val==np.nan)):
                            t1_dict[val].append(i)
                    for j, (_, row2) in enumerate(table2.iterrows()):
                        val = row2[attrib2]
                        if not (val == None or (type(val)==str and val == '') or (type(val)==float and val==np.nan)):
                            t2_dict[val].append(j)
                    t1_values = [k for k, _ in t1_dict.items()]
                    t2_values = [k for k, _ in t2_dict.items()]
                    fp_join_sizeU = 0
                    fp_join_sizeB = 0
                    fp_join_sizeF = 0
                    non_empty = False 
                    
                    # Computes the join by taking the cartesian product of the rows with equal values
                    # from the two tables
                    for t1_value in t1_values:
                        if t1_value in t2_values:
                            non_empty = True
                            
                            # Updates the FP bandwidth estimate
                            if statistics == "stats":
                                fp_join_sizeU += (uniform(tables, file1, attrib1)) * (uniform(tables, file2, attrib2))
                                fp_join_sizeB += bucketHist(bHist1, t1_value) * bucketHist(bHist2, t1_value)
                                fp_join_sizeF += fullHist(tables, file1, attrib1, t1_value) * fullHist(tables, file2, attrib2, t1_value)
                            
                            # Updates the connected tables
                            for i in t1_dict[t1_value]:
                                for _ in range(len(t2_dict[t1_value])):
                                    for conn_file in conn_group1:
                                        new_inner_fp[conn_file] = new_inner_fp[conn_file].append(innerTables1[conn_file].iloc[i])
                            for j in t2_dict[t1_value]:
                                for _ in range(len(t1_dict[t1_value])):
                                    for conn_file in conn_group2:
                                        new_inner_fp[conn_file] = new_inner_fp[conn_file].append(innerTables2[conn_file].iloc[j])
                    if not non_empty:
                        return None
                    # Finishes updating the statistics
                    if statistics == "stats":
                        max_join_sizeU = innerBEstU1[file1] * innerBEstU2[file2]
                        max_join_sizeB = innerBEstB1[file1] * innerBEstB2[file2]
                        max_join_sizeF = innerBEstF1[file1] * innerBEstF2[file2]
                        for conn_file in conn_group1:
                            innerBEstFPU[conn_file] = fp_join_sizeU * max_join_sizeU
                            innerBEstFPB[conn_file] = fp_join_sizeB * max_join_sizeB
                            innerBEstFPF[conn_file] = fp_join_sizeF * max_join_sizeF
                        for conn_file in conn_group2:
                            innerBEstFPU[conn_file] = fp_join_sizeU * max_join_sizeU
                            innerBEstFPB[conn_file] = fp_join_sizeB * max_join_sizeB
                            innerBEstFPF[conn_file] = fp_join_sizeF * max_join_sizeF
                            
                    # Updates the connections
                    new_connections = connections_temp1 + connections_temp2
                    new_connections.append(new_conn_group)
                
                # If we are not only computing FP joins, add PP results
                if statistics != "FP":
                    # Copies the tables
                    new_inner_pp = deepcopy(innerTables1)
                    new_inner_pp.update(innerTables2)
                    for f in new_conn_group:
                        new_inner_pp[f] = pd.DataFrame(columns=new_inner_pp[f].columns)
                    t1_set = set()
                    t2_set = set()
                    t1 = tables[file1][attrib1]
                    t2 = tables[file2][attrib2]
                    
                    # Creates sets of which values are needed to compute the join
                    # clientside
                    for val in t1:
                        if not (val == None or (type(val)==str and val == '') or (type(val)==float and val==np.nan)):
                            t1_set.add(val)
                    for val in t2:
                        if not (val == None or (type(val)==str and val == '') or (type(val)==float and val==np.nan)):
                            t2_set.add(val)
                    
                    # Returns None if the join is empty
                    if t1_set.isdisjoint(t2_set):
                        return None
                    keep_values = t1_set.union(t2_set)
                    # Filters the tables to the rows that have values in the intersection
                    # of the two ranges
                    for i, (_, row1) in enumerate(table1.iterrows()):
                        if row1[attrib1] in keep_values:
                            for conn_file in conn_group1:
                                new_inner_pp[conn_file] = new_inner_pp[conn_file].append(innerTables1[conn_file].iloc[i])
                    for j, (_, row2) in enumerate(table2.iterrows()):
                        if row2[attrib2] in keep_values:
                            for conn_file in conn_group2:
                                new_inner_pp[conn_file] = new_inner_pp[conn_file].append(innerTables2[conn_file].iloc[j])
                    
                    # Updates the statistics
                    if statistics == "stats":
                        t1_access_proportionU = 0
                        t2_access_proportionU = 0
                        t1_access_proportionB = 0
                        t2_access_proportionB = 0
                        t1_access_proportionF = 0
                        t2_access_proportionF = 0
                        for val in t1_set.union(t2_set):
                            t1_access_proportionU += uniform(tables, file1, attrib1)
                            t2_access_proportionU += uniform(tables, file2, attrib2)
                            t1_access_proportionB += bucketHist(bHist1, val)
                            t2_access_proportionB += bucketHist(bHist2, val)
                            t1_access_proportionF += fullHist(tables, file1, attrib1, val)
                            t2_access_proportionF += fullHist(tables, file2, attrib2, val)
                        for conn_file in conn_group1:
                            innerBEstPPU[conn_file] *= t1_access_proportionU
                            innerBEstPPB[conn_file] *= t1_access_proportionB
                            innerBEstPPF[conn_file] *= t1_access_proportionF
                        for conn_file in conn_group2:
                            innerBEstPPU[conn_file] *= t1_access_proportionU
                            innerBEstPPB[conn_file] *= t1_access_proportionB
                            innerBEstPPF[conn_file] *= t1_access_proportionF
                    
                    # Updates the connections
                    new_connections = connections1 + connections2
                
                # Adds the new possible results with an updated join annotation string
                if statistics != "PP":
                    newResults.append((new_inner_fp, new_connections, (innerBEstFPU, innerBEstFPB, innerBEstFPF),"1" + retString1 + retString2)) # old L: innerL1 + innerL2 + len(t1_set.union(t2_set))
                if statistics != "FP":
                    newResults.append((new_inner_pp, new_connections, (innerBEstPPU, innerBEstPPB, innerBEstPPF), "0" + retString1 + retString2)) # old L: innerL1 + innerL2 + 2
        # Returns the new results. Note that the size of newResults is exponential in the
        # number of joins in the query but we heuristically assume this to be small
        return newResults
    
    # Handles file selection
    else:
        file = operation
        total_table_rows += tables[file].shape[0]
        rt = {file : tables[file].shape[0]}
        
        # Returns the file with the only connection being itself, the bandwidth
        # estimates set to that file's number of rows and an empty join annotation
        return [({file : tables[file]}, [[file]], (rt, deepcopy(rt), deepcopy(rt)), "")]


# Runs the category of queries with the specified number of joins and number of
# selects. Stores the results in a pickle file
#   tables : Dict - Maps relation name Strings to Dataframes
#   js : Int - The desired number of joins in the query
#   ss : Int - The desired number of selects in the query
def runSJ(tables, js, ss):
    global total_table_rows
    
    # Reads in the queries
    queries = pickle.load( open( "joins" + str(js) + "Selects" + str(ss) + ".p", "rb" ) )
    qrys = [qry for (qry, b, values) in queries]
    values = [value for (qry, b, value) in queries]
    
    # Prepares to stores information
    total = 0
    totalBestNumPP = 0
    summary = {}
    summary["sj"] = []
    wrong1U, wrong2U, wrong3U, correctU = 0, 0, 0, 0
    wrong1B, wrong2B, wrong3B, correctB = 0, 0, 0, 0
    wrong1F, wrong2F, wrong3F, correctF = 0, 0, 0, 0
    
    # For each query
    for i in range(len(qrys)):
        qry = qrys[i]
        v = values[i]
        total_table_rows = 0
        
        # Computes the query results for each possible join annotation with each
        # heuristic histogram option
        results = getQueryTables(qry, tables, v, statistics="stats")
        bandwidthFP = 0
        bandwidthPP = 0
        minBandwidth = 0
        threshold = 0
        thresh_count = 0
        
        # For each result, gathers info to set the client bandwidth threshold
        for res in results:
            thresh_count += 1
            (resTables, resConnections, resEstimates, resString) = res
            resTables = [table for _, table in resTables.items()]
            for t in resTables:
                threshold += t.shape[0]
        
        # Sets the heuristic threshold to the average of the bandwidths returned
        # by the annotation choices. This means that there will always be at least
        # one valid query below or equal to the bandwidth threshold.
        # This is not entirely realistic because in many cases, all annotations may
        # be well below a client's realistic bandwidth threshold. However, it matches
        # up with our assumption that a client will not intentionally query for more rows
        # than they can store.
        threshold = threshold / thresh_count
        
        
        trueBestBand = 100000000000000000000000
        estForTrueU = 0
        estForTrueB = 0
        estForTrueF = 0
        bestNumPP = 0
        bestNumPPU = 0
        bestNumPPB = 0
        bestNumPPF = 0
        bestBandU = 0
        bestBandB = 0
        bestBandF = 0
        
        # For each result, check the success or reason for failure for each bandwidth
        # estimate with the 3 histogram options
        for res in results:
            total += 1
            (resTables, resConnections, resEstimates, resString) = res
            bandwidth = 0
            estBandwidthU = 0
            estBandwidthB = 0
            estBandwidthF = 0
            (resEstimatesU, resEstimatesB, resEstimatesF) = resEstimates
            resTables = [table for _, table in resTables.items()]
            resEstimatesU = [est for _, est in resEstimatesU.items()]
            resEstimatesB = [est for _, est in resEstimatesB.items()]
            resEstimatesF = [est for _, est in resEstimatesF.items()]
            for t in resTables:
                bandwidth += t.shape[0]
            for est in resEstimatesU:
                estBandwidthU += est
            for est in resEstimatesB:
                estBandwidthB += est
            for est in resEstimatesF:
                estBandwidthF += est
            if stringIsAll(resString, "1"):
                bandwidthFP = bandwidth
                for t in resTables:
                    minBandwidth += t.drop_duplicates().shape[0]
            if stringIsAll(resString, "0"):
                bandwidthPP = bandwidth
            type = ""
            numPP = 0
            for c in resString:
                if c=="0":
                    numPP += 1
                    type += "PP, "
                if c=="1":
                    type += "FP, "
            
            # If the bandwidth is below the threshold and there are more PP joins
            # or equal PP joins and a smaller bandwidth, set this annotation to be
            # the true best annotation
            if bandwidth < threshold:
                if numPP > bestNumPP:
                    bestNumPP = numPP
                    trueBestBand = bandwidth
                    estForTrueU = estBandwidthU
                    estForTrueB = estBandwidthB
                    estForTrueF = estBandwidthF
                elif numPP == bestNumPP and trueBestBand > bandwidth:
                    trueBestBand = bandwidth
                    estForTrueU = estBandwidthU
                    estForTrueB = estBandwidthB
                    estForTrueF = estBandwidthF
            
            # For uniform, bucketed, and full
            # If the estimated bandwidth is below the threshold and there are more PP joins
            # or equal PP joins and a smaller estimated bandwidth, set this annotation to be
            # the best annotation for that histogram option
            if estBandwidthU < threshold:
                if numPP > bestNumPPU:
                    bestNumPPU = numPP
                    bestBandU = estBandwidthU
                elif numPP == bestNumPPU and bestBandU > estBandwidthU:
                    bestBandU = estBandwidthU
            if estBandwidthB < threshold:
                if numPP > bestNumPPB:
                    bestNumPPB = numPP
                    bestBandB = estBandwidthB
                elif numPP == bestNumPPB and bestBandB > estBandwidthB:
                    bestBandB = estBandwidthB
            if estBandwidthF < threshold:
                if numPP > bestNumPPF:
                    bestNumPPF = numPP
                    bestBandF = estBandwidthF
                elif numPP == bestNumPPF and bestBandF > estBandwidthF:
                    bestBandF = estBandwidthF
                    
            # Prints information about the query
            print(qry)
            print("JOIN types: " + type)
            print("Bandwidth: " + str(bandwidth))
            print("Estimated bandwidth uniform: " + str(estBandwidthU))
            print("Estimated bandwidth bucket: " + str(estBandwidthB))
            print("Estimated bandwidth full: " + str(estBandwidthF))
            print("------------------------")
        
        # For uniform, bucketed, and full
        # Using the true best annotation and the heuristic best annotations,
        # Update the number of queries which are correct and wrong both the different
        # categories
        totalBestNumPP += bestNumPP
        if estForTrueU > threshold:
            wrong1U += 1
        elif bestNumPPU > bestNumPP:
            wrong2U += 1
        elif estForTrueU != bestBandU:
            wrong3U += 1
        else:
            correctU += 1
        if estForTrueB > threshold:
            wrong1B += 1
        elif bestNumPPB > bestNumPP:
            wrong2B += 1
        elif estForTrueB != bestBandB:
            wrong3B += 1
        else:
            correctB += 1
        if estForTrueF > threshold:
            wrong1F += 1
        elif bestNumPPF > bestNumPP:
            wrong2F += 1
        elif estForTrueF != bestBandF:
            wrong3F += 1
        else:
            correctF += 1
            
        # Add to a list of 3-tuples of the ratios of all fp and all pp bandwidth
        # to the min bandwidth needed to compute the join under the "sj" key
        summary["sj"].append((bandwidthFP/minBandwidth, bandwidthPP/minBandwidth, minBandwidth))
        print("Fp scale from min: " + str(bandwidthFP/minBandwidth))
        print("Pp scale from min: " + str(bandwidthPP/minBandwidth))
    
    # Print heuristic accuracy information
    print(wrong1U)
    print(wrong2U)
    print(wrong3U)
    print(correctU)
    print(wrong1B)
    print(wrong2B)
    print(wrong3B)
    print(correctB)
    print(wrong1F)
    print(wrong2F)
    print(wrong3F)
    print(correctF)
    print(totalBestNumPP/total)
    
    # Add the heuristic accuracy information to the summary under "heuristic"
    summary["heuristic"] = {"U" : (wrong1U, wrong2U, wrong3U, correctU), "B" : (wrong1B, wrong2B, wrong3B, correctB), "F" : (wrong1F, wrong2F, wrong3F, correctF) }
    
    # Store the summary in a pickle file named "resultsJ[# joins in query]S[# selects in query].p"
    print("resultsJ" + str(js) + "S" + str(ss) + ".p")
    pickle.dump(summary, open("resultsJ" + str(js) + "S" + str(ss) + ".p", "wb" ) )


# Function to generate random  queries with a certain number of  joins and selects
# from either the Chicago or Sakila database.
#
# Note that for the sake of computational time, this process was distributed to
# be done independently on multiple computers and then the resulting queries were
# merged together into one group. Therefore, much of this function was designed
# to only run some fraction of the query generation while the remaining parts were
# commented out and running on different machines.
# I'm restoring the function to generate all queries, but keep in mind that it was
# split up into multiple parallel runs to generate the results given in the paper.
# Also, the three branches of the main "if" statement for 1, 2, and 3 joins could
# likely be joined into a single helper function with far better code reuse and less
# copy and pasting, but the current design is result of growth through necessity rather
# than careful planning.  
#
# Stores queries in pickle files of the format "joins[# of joins]Selects[# of selects].p"
# Note: 1 join and 0 selects was done separately
#   tables : Dict - Dictionary mapping relation name Strings to Dataframes
def getQueries(tables):
    # Opens the pickle files to see what progress has already been made towards
    # generating the queries of the different types
    j1s1 = pickle.load( open("joins1Selects1.p", "rb" ) )
    print(len(j1s1))
    j1s2 = pickle.load( open("joins1Selects2.p", "rb" ) )
    print(len(j1s2))
    j2s0 = pickle.load( open("joins2Selects0.p", "rb" ) )
    print(len(j2s0))
    j2s1 = pickle.load( open("joins2Selects1.p", "rb" ) )
    print(len(j2s1))
    j2s2 = pickle.load( open("joins2Selects2.p", "rb" ) )
    print(len(j2s2))
    j3s2 = pickle.load( open("joins3Selects2.p", "rb" ) )
    print(len(j3s2))
    j3s1 = pickle.load( open("joins3Selects1.p", "rb" ) )
    print(len(j3s1))
    j3s0 = pickle.load( open("joins3Selects0.p", "rb" ) )
    print(len(j3s0))
    
    # # Defunct. Code to reset the pickle files
    # # ------------------------------------------------------
    # pickle.dump([], open("joins1Selects1.p", "wb" ) )
    # pickle.dump([], open("joins1Selects2.p", "wb" ) )
    # pickle.dump([], open("joins2Selects0.p", "wb" ) )
    # pickle.dump([], open("joins2Selects1.p", "wb" ) )
    # pickle.dump([], open("joins2Selects2.p", "wb" ) )
    # pickle.dump([], open("joins3Selects0.p", "wb" ) )
    # pickle.dump([], open("joins3Selects1.p", "wb" ) )
    # pickle.dump([], open("joins3Selects2.p", "wb" ) )
    # quit()
    # # ------------------------------------------------------
    
    
    # Opens up pickle files containing all non-empty joins from the Sakila and
    # Chicago databases. These are Dicts mapping each attribute to a list of the attributes
    # that they can be joined with for a non-empty relation
    sakila_joins = pickle.load( open( "sakila_joins.p", "rb" ) )
    chicago_joins = pickle.load( open( "chicago_joins.p", "rb" ) )
    sakila_keys = [k for k, _ in sakila_joins.items()]
    chicago_keys = [k for k, _ in chicago_joins.items()]
    keys = sakila_keys + chicago_keys
    
    # # For 1 join and 1 or 2 selects
    for i in [2, 1]:
        queries = pickle.load( open("joins1Selects" + str(i) + ".p", "rb" ) )
        
        # Continues generating queries until there are 25
        while len(queries) < 25:
            values = {}
            
            # Picks an attribute which has a joinable attribute in the database at random
            file_attrib1 = random.choice(keys)
            
            # Chooses the attribute to join with at random from the dictionary of
            # non-empty joines
            if file_attrib1 in sakila_keys:
                file_attrib2 = random.choice(sakila_joins[file_attrib1])
            if file_attrib1 in chicago_keys:
                file_attrib2 = random.choice(chicago_joins[file_attrib1])
            
            # Parses out the information to format as a query
            file1 = file_attrib1.split(".")[0]
            file2 = file_attrib2.split(".")[0]
            fileQ1 = "(" + file1 + ")"
            fileQ2 = "(" + file2 + ")"
            
            # Creates subqueries with 1 or 2 selections
            for j in range(i):
                # Chooses a random attribute from one of the two tables and then applies
                # the selection to that table
                attribSel = random.choice(list(tables[file1].columns) + list(tables[file2].columns))
                if attribSel in list(tables[file1].columns):
                    val = random.choice(tables[file1][attribSel].unique())
                    values["vSel" + str(j)] = val
                    fileQ1 = "(SELECT (" + file1 + "." + attribSel + "=vSel" + str(j) + ") " + fileQ1 + ")"
                elif attribSel in list(tables[file2].columns):
                    val = random.choice(tables[file2][attribSel].unique())
                    values["vSel" + str(j)] = val
                    fileQ2 = "(SELECT (" + file2 + "." + attribSel + "=vSel" + str(j) + ") " + fileQ2 + ")"
            
            # Finishes generating the query
            qry = "(JOIN (" + file_attrib1 + "=" + file_attrib2 + ") " + fileQ1 + " " + fileQ2 + ")"
            print(qry)
            
            # Actually computes the query, using FP joins so nothing is left as post-processing
            resultsFP = getQueryTables(qry, tables, values, statistics="FP")
            bandwidthFP = 0
            
            # If the query is non-empty, add it to the list of queries for this category
            # and store the results for re-use
            if resultsFP == None:
                print("Fail")
                continue
            print("Success!")
            for res in resultsFP:
                (resTablesFP, _, _, _) = res
            resTablesFP = [table for _, table in resTablesFP.items()]
            for t in resTablesFP:
                bandwidthFP += t.shape[0]
            queries.append((qry, bandwidthFP, values))
            pickle.dump(queries, open("joins1Selects" + str(i) + ".p", "wb" ) )
            
    # Repeats the process for two joins and 0, 1, or 2 selects
    # Uses the same idea as above but now must join twice
    for i in [0, 1, 2]:
        queries = pickle.load( open("joins2Selects" + str(i) + ".p", "rb" ) )
        while len(queries) < 25:
            values = {}
            
            # Gets a possible option for the first join attribute
            file_attrib1 = random.choice(keys)
            
            # Gets the second join attribute and then starts spinning for new attributes
            # to join with for the second join
            if file_attrib1 in sakila_keys:
                file_attrib2 = random.choice(sakila_joins[file_attrib1])
                file2 = file_attrib2.split(".")[0]
                file_attrib22 = file2 + "." + random.choice(list(tables[file2].columns))
                attempt = 0
                while file_attrib22 not in sakila_keys:
                    file_attrib22 = file2 + "." + random.choice(list(tables[file2].columns))
                    attempt += 1
                    if attempt > 100:
                        break
                if attempt > 100:
                    continue
                file_attrib3 = random.choice(sakila_joins[file_attrib22])
            if file_attrib1 in chicago_keys:
                file_attrib2 = random.choice(chicago_joins[file_attrib1])
                file2 = file_attrib2.split(".")[0]
                file_attrib22 = file2 + "." + random.choice(list(tables[file2].columns))
                attempt = 0
                while file_attrib22 not in chicago_keys:
                    file_attrib22 = file2 + "." + random.choice(list(tables[file2].columns))
                    attempt += 1
                    if attempt > 100:
                        break
                if attempt > 100:
                    continue
                file_attrib3 = random.choice(chicago_joins[file_attrib22])
            
            # Extracts the information to form a query
            file1 = file_attrib1.split(".")[0]
            file2 = file_attrib2.split(".")[0]
            file3 = file_attrib3.split(".")[0]
            if file1 == file3 or tables[file1].shape[0] > 1000 or tables[file2].shape[0] > 1000 or tables[file3].shape[0] > 1000:
                continue
            fileQ1 = "(" + file1 + ")"
            fileQ2 = "(" + file2 + ")"
            fileQ3 = "(" + file3 + ")"
            
            # Applies 0, 1, or 2 selects
            for j in range(i):
                attribSel = random.choice(list(tables[file1].columns) + list(tables[file2].columns) + list(tables[file3].columns))
                if attribSel in list(tables[file1].columns):
                    val = random.choice(tables[file1][attribSel].unique())
                    values["vSel" + str(j)] = val
                    fileQ1 = "(SELECT (" + file1 + "." + attribSel + "=vSel" + str(j) + ") " + fileQ1 + ")"
                elif attribSel in list(tables[file2].columns):
                    val = random.choice(tables[file2][attribSel].unique())
                    values["vSel" + str(j)] = val
                    fileQ2 = "(SELECT (" + file2 + "." + attribSel + "=vSel" + str(j) + ") " + fileQ2 + ")"
                elif attribSel in list(tables[file3].columns):
                    val = random.choice(tables[file3][attribSel].unique())
                    values["vSel" + str(j)] = val
                    fileQ3 = "(SELECT (" + file3 + "." + attribSel + "=vSel" + str(j) + ") " + fileQ3 + ")"
                    
            # Finishes putting together the query
            qry = "(JOIN (" + file_attrib3 + "=" + file_attrib22 + ") " + fileQ3 + " (JOIN (" + file_attrib1 + "=" + file_attrib2 + ") " + fileQ1 + " " + fileQ2 + "))"
            print(qry)
            
            # Actually tests the query
            resultsFP = getQueryTables(qry, tables, values, statistics="FP")
            bandwidthFP = 0
            
            # Keeps searching if the query is empty
            if resultsFP == None:
                print("Fail")
                continue
            
            # Otherwise, adds the query to the list for the category
            print("Success!")
            for res in resultsFP:
                (resTablesFP, _, _, _) = res
            resTablesFP = [table for _, table in resTablesFP.items()]
            for t in resTablesFP:
                bandwidthFP += t.shape[0]
            queries.append((qry, bandwidthFP, values))
            pickle.dump(queries, open("joins2Selects" + str(i) + ".p", "wb" ) )
    
    # Repeats the process for three joins and 0, 1, or 2 selects
    # Uses the same idea as above but now must join thrice
    for i in [0, 1, 2]:
        queries = pickle.load( open("joins3Selects" + str(i) + ".p", "rb" ) )
        while len(queries) < 25:
            values = {}
            
            # Gets the joins
            file_attrib1 = random.choice(keys)
            if file_attrib1 in sakila_keys:
                file_attrib2 = random.choice(sakila_joins[file_attrib1])
                file2 = file_attrib2.split(".")[0]
                file_attrib22 = file2 + "." + random.choice(list(tables[file2].columns))
                attempt = 0
                while file_attrib22 not in sakila_keys:
                    file_attrib22 = file2 + "." + random.choice(list(tables[file2].columns))
                    attempt += 1
                    if attempt > 100:
                        break
                if attempt > 100:
                    continue
                file_attrib3 = random.choice(sakila_joins[file_attrib22])
                file3 = file_attrib3.split(".")[0]
                file_attrib32 = file3 + "." + random.choice(list(tables[file3].columns))
                attempt = 0
                while file_attrib32 not in sakila_keys:
                    file_attrib32 = file3 + "." + random.choice(list(tables[file3].columns))
                    attempt += 1
                    if attempt > 100:
                        break
                if attempt > 100:
                    continue
                file_attrib4 = random.choice(sakila_joins[file_attrib32])
            if file_attrib1 in chicago_keys:
                file_attrib2 = random.choice(chicago_joins[file_attrib1])
                file2 = file_attrib2.split(".")[0]
                file_attrib22 = file2 + "." + random.choice(list(tables[file2].columns))
                attempt = 0
                while file_attrib22 not in chicago_keys:
                    file_attrib22 = file2 + "." + random.choice(list(tables[file2].columns))
                    attempt += 1
                    if attempt > 100:
                        break
                if attempt > 100:
                    continue
                file_attrib3 = random.choice(chicago_joins[file_attrib22])
                file3 = file_attrib3.split(".")[0]
                file_attrib32 = file3 + "." + random.choice(list(tables[file3].columns))
                attempt = 0
                while file_attrib32 not in chicago_keys:
                    file_attrib32 = file3 + "." + random.choice(list(tables[file3].columns))
                    attempt += 1
                    if attempt > 100:
                        break
                if attempt > 100:
                    continue
                file_attrib4 = random.choice(chicago_joins[file_attrib32])
            
            #  Prepares to format the tables for a query
            file1 = file_attrib1.split(".")[0]
            file2 = file_attrib2.split(".")[0]
            file3 = file_attrib3.split(".")[0]
            file4 = file_attrib4.split(".")[0]
            if file1 == file3 or file1 == file4 or file2 == file4 or tables[file1].shape[0] > 1000 or tables[file2].shape[0] > 1000 or tables[file3].shape[0] > 1000 or tables[file4].shape[0] > 1000:
                continue
            fileQ1 = "(" + file1 + ")"
            fileQ2 = "(" + file2 + ")"
            fileQ3 = "(" + file3 + ")"
            fileQ4 = "(" + file4 + ")"
            
            # Prepares the selects
            for j in range(i):
                attribSel = random.choice(list(tables[file1].columns) + list(tables[file2].columns) + list(tables[file3].columns) + list(tables[file4].columns))
                if attribSel in list(tables[file1].columns):
                    val = random.choice(tables[file1][attribSel].unique())
                    values["vSel" + str(j)] = val
                    fileQ1 = "(SELECT (" + file1 + "." + attribSel + "=vSel" + str(j) + ") " + fileQ1 + ")"
                elif attribSel in list(tables[file2].columns):
                    val = random.choice(tables[file2][attribSel].unique())
                    values["vSel" + str(j)] = val
                    fileQ2 = "(SELECT (" + file2 + "." + attribSel + "=vSel" + str(j) + ") " + fileQ2 + ")"
                elif attribSel in list(tables[file3].columns):
                    val = random.choice(tables[file3][attribSel].unique())
                    values["vSel" + str(j)] = val
                    fileQ3 = "(SELECT (" + file3 + "." + attribSel + "=vSel" + str(j) + ") " + fileQ3 + ")"
                elif attribSel in list(tables[file4].columns):
                    val = random.choice(tables[file4][attribSel].unique())
                    values["vSel" + str(j)] = val
                    fileQ4 = "(SELECT (" + file4 + "." + attribSel + "=vSel" + str(j) + ") " + fileQ4 + ")"
            
            # Finishes the query
            qry = "(JOIN (" + file_attrib4 + "=" + file_attrib32 + ") " + fileQ4 + " (JOIN (" + file_attrib3 + "=" + file_attrib22 + ") " + fileQ3 + " (JOIN (" + file_attrib1 + "=" + file_attrib2 + ") " + fileQ1 + " " + fileQ2 + ")))"
            print(qry)
            
            # Tests the query
            resultsFP = getQueryTables(qry, tables, values, statistics="FP")
            bandwidthFP = 0
            
            # Stores the query depending on the results
            if resultsFP == None:
                print("Fail")
                continue
            print("Success!")
            for res in resultsFP:
                (resTablesFP, _, _, _) = res
            resTablesFP = [table for _, table in resTablesFP.items()]
            for t in resTablesFP:
                bandwidthFP += t.shape[0]
            queries.append((qry, bandwidthFP, values))
            pickle.dump(queries, open("joins3Selects" + str(i) + ".p", "wb" ) )


# Function that prints a summary of our hybrid heuristic experiments
def printSummary():
    # Loads the results of the queries generated by runSJ. There are 25 queries in
    # each category. Results placed in pickle files to save time running the same
    # queries repeatedly.
    # Note: Results from one join, 0 select queries done separately.
    # Format is j[# joins in the query]s[# selects in the query]
    j1s1 = pickle.load( open( "resultsJ1S1.p", "rb" ) )
    j1s2 = pickle.load( open( "resultsJ1S2.p", "rb" ) )
    j2s0 = pickle.load( open( "resultsJ2S0.p", "rb" ) )
    j2s1 = pickle.load( open( "resultsJ2S1.p", "rb" ) )
    j2s2 = pickle.load( open( "resultsJ2S2.p", "rb" ) )
    j3s0 = pickle.load( open( "resultsJ3S0.p", "rb" ) )
    j3s1 = pickle.load( open( "resultsJ3S1.p", "rb" ) )
    j3s2 = pickle.load( open( "resultsJ3S2.p", "rb" ) )
    
    # Prints the possible categories of heuristic failure
    print("Failure reason 1 = the estimate for the true best was over the threshold")
    print("Failure reason 2 = the estimate for the true best was below the threshold, but it chose with more PP incorrectly")
    print("Failure reason 3 = the estimate for the true best was below the threshold, but it chose with equal PP incorrectly")
    summaries = [j1s1, j1s2, j2s0, j2s1, j2s2, j3s0, j3s1, j3s2]
    
    # For the summary of each group of 25 queries. Each summary is a Dict
    for summary in summaries:
        # Extracts statistics
        sj = summary["sj"]
        pp_ratios = [p for (_, p, _) in sj]
        fp_ratios = [f for (f, _, _) in sj]
        ideal = [min for (_, _, min) in sj]
        
        min_pp = min(pp_ratios)
        min_fp = min(fp_ratios)
        min_ideal = min(ideal)
        mean_pp = sum(pp_ratios)/len(pp_ratios)
        mean_fp = sum(fp_ratios)/len(fp_ratios)
        mean_ideal = sum(ideal)/len(ideal)
        var_pp = np.var(pp_ratios)
        var_fp = np.var(fp_ratios)
        var_ideal = np.var(ideal)
        max_pp = max(pp_ratios)
        max_fp = max(fp_ratios)
        max_ideal = max(ideal)
        
        heuristicU = summary["heuristic"]["U"]
        heuristicB = summary["heuristic"]["B"]
        heuristicF = summary["heuristic"]["F"]
        
        # Prints statistics
        print("Min/Mean/Max # ideal rows: " + str(min_ideal) + "/" + str(mean_ideal) + "/" + str(max_ideal))
        print("Variance # ideal rows: " + str(var_ideal))
        print("Min/Mean/Max ratio of pp to ideal: " + str(min_pp) + "/" + str(mean_pp) + "/" + str(max_pp))
        print("Variance ratio of pp to ideal: " + str(var_pp))
        print("Min/Mean/Max ratio of fp to ideal: " + str(min_fp) + "/" + str(mean_fp) + "/" + str(max_fp))
        print("Variance ratio of fp to ideal: " + str(var_fp))
        
        print("")
        
        # Prints heuristic success rate for each histogram type and for each
        # failure category
        print("Number of times uniform failed r1")
        print(heuristicU[0])
        print("Number of times uniform failed r2")
        print(heuristicU[1])
        print("Number of times uniform failed r3")
        print(heuristicU[2])
        print("Number of times uniform correct")
        print(heuristicU[3])
        
        print("Number of times buckets failed r1")
        print(heuristicB[0])
        print("Number of times buckets failed r2")
        print(heuristicB[1])
        print("Number of times buckets failed r3")
        print(heuristicB[2])
        print("Number of times buckets correct")
        print(heuristicB[3])
        
        print("Number of times full failed r1")
        print(heuristicF[0])
        print("Number of times full failed r2")
        print(heuristicF[1])
        print("Number of times full failed r3")
        print(heuristicF[2])
        print("Number of times full correct")
        print(heuristicF[3])


# ------------------- HYBRID HEURISTIC HELPERS -------------

# Computes the estimated percentage of a select using full histograms
#   tables : Dict - Dictionary from relation name Strings to pandas dataframes
#   file : String - the name of the relation for the select
#   attrib : String - the name of the attribute for the select
#   value : String - the name of the selected value
#   Returns the proportion of the table equal to the value at that attribute with
#       full histograms
def fullHist(tables, file, attrib, value):
    return len(list(np.where(tables[file][attrib] == value)[0]))/tables[file].shape[0]


# Constructs bucketed histograms from an attribute with a hardcoded size of 200
#   tables : Dict - Dictionary from relation name Strings to pandas dataframes
#   file : String - the name of the relation for the histograms
#   attrib : String - the name of the attribute for the histograms
#   Returns a Histogram, a pair with the histogram and the bucket edges
def makeBucketHist(tables, file, attrib):
    return 0
    bucketNum = min(200, len(tables[file][attrib].unique()))
    buckets = [(1/bucketNum) * i for i in range(bucketNum)]
    bin_edges = stats.mstats.mquantiles([toInt(v) for v in tables[file][attrib]], buckets)
    # print(bin_edges)
    # print("------")
    hist = np.histogram([toInt(v) for v in tables[file][attrib]], bins=bin_edges)
    (h, edges) = hist
    h = [histVal / tables[file].shape[0] for histVal in h]
    return (h, edges)


# Computes the estimated percentage of a select assuming bucketed histograms
#   hist : Histogram - A bucketed histogram (a pair with the histogram and the bucket)
#       edges
#   value : String - the value for the select
#   Returns the proportion of the table equal to the value at that attribute with
#       bucketed histograms
def bucketHist(hist, value):
    return 0
    (h, edges) = hist
    # print("H len: " + str(len(h)))
    # print("Edges len: " + str(len(edges)))
    for i in range(min(len(edges),len(h))):
        if toInt(value) <= edges[i]:
            # print(h[min(i, len(edges)-2)])
            # print("Bucket Hist returning: " + str(h[min(i, len(edges)-2)] * (1 / (edges[i] - edges[i-1]))))
            return h[min(i, len(edges)-2)] * (1 / (edges[i] - edges[i-1]))
    return 0


# Computes the estimated percentage of a select assuming uniformity
#   tables : Dict - Dictionary from relation name Strings to pandas dataframes
#   file : String - the name of the relation for the select
#   attrib : String - the name of the attribute for the select
#   Returns the proportion of the table equal to the value at that attribute with
#       an assumption of uniformity (since we assume uniformity, it does not
#       need to know the value selected)
def uniform(tables, file, attrib):
    # print("Uniform returning: " + str(1/len(tables[file][attrib].unique())))
    return 1/len(tables[file][attrib].unique())


# ------------------- MISCELLANEOUS HELPERS ------------------------
# Helper function to check if a string is all one repeating character. Used for telling
# when an annotated query is all fp or all pp
#   s : String - a string
#   c : String - a single character
#   Returns a Bool
def stringIsAll(s, c):
    for ch in s:
        if ch != c:
            return False
    return True


# Helper function for converting an ascii hex string into an Int
#   v : String - A string
#   Returns an Int
def toInt(v):
    if type(v)==str:
        # print(v)
        hex = "".join([format(ord(c), "x") for c in v])
        return int(hex, 16)
    else:
        return v


# Helper function that merges two dictionaries mapping Strings to Ints, summing the
# two values where the strings are equal
#   join_dict1 : Dict
#   join_dict2 : Dict
#   Returns a Dict
def mergeJoinDicts(join_dict1, join_dict2):
    new_dict = {}
    d1keys = [k for k, _ in join_dict1.items()]
    d2keys = [k for k, _ in join_dict2.items()]
    for key in d1keys:
        if key in d2keys:
            new_dict[key] = join_dict1[key] + join_dict2[key]
        else:
            new_dict[key] = join_dict1[key]
    new_keys = [k for k, _ in new_dict.items()]
    for key in d2keys:
        if not key in new_keys:
            new_dict[key] = join_dict2[key]
    pickle.dump(new_dict, open("chicago_joins.p", "wb" ) )


# ------------------- MAIN ------------------------


# Prints the test harness commands
def printCommands():
    print("Here are the simulation commands:")
    print("help         - prints the commands")
    print("loadTables   - loads in the Sakila and Chicago datasets")
    print("tableInfo    - gets info on the loaded datasets")
    print("singleJoins  - runs experiments on single attribute joins")
    print("getNonEmpty  - creates pickle files with the non-empty single attribute joins. Needed to generate queries.")
    print("genQueries   - generates 200 random queries from the Sakila and Chicago datasets")
    print("runQueries   - generates a summary from the 200 random queries")
    print("printSummary - prints a summary of the results from running 200 queries")
    print("q            - quits the test harness")


# The main function
def main():
    print("Welcome! The code here is used to replicate the simulations in https://eprint.iacr.org/2021/852.pdf")
    printCommands()
    quit = False
    tables = {}
    sak_file_names = []
    chi_file_names = []
    while not quit:
        command = input("> ")
        # Quits the test harness
        if command == "q":
            quit = True
        
        # Prints the test harness commands
        elif command == "help":
            printCommands()
            
        # Prints the hybrid heuristic summary
        elif command == "printSummary":
            printSummary()
        
        # Runs the 200 queries needed to generate the hybrid heuristic summary
        elif command == "runQueries":
            runSJ(tables, 1, 1)
            runSJ(tables, 1, 2)
            runSJ(tables, 2, 0)
            runSJ(tables, 2, 1)
            runSJ(tables, 2, 2)
            runSJ(tables, 3, 0)
            runSJ(tables, 3, 1)
            runSJ(tables, 3, 2)
        
        # Generates 200 random queries
        elif command == "genQueries":
            getQueries(tables)
        
        # Loads the Chicago and Sakila data
        elif command == "loadTables":
            chi_file_names = os.listdir('chicago15')
            chi_file_names = list(filter(lambda x : not x == '.DS_Store', chi_file_names))
            sak_file_names = os.listdir('sakila-db/sakila-csv')
            sak_file_names = list(filter(lambda x : not x == '.DS_Store', sak_file_names))
            tables = {}
            
            for fName in chi_file_names + sak_file_names:
                if not fName == ".DS_Store":
                    if fName in chi_file_names:
                        table = pd.read_csv("chicago15/" + fName)
                    if fName in sak_file_names:
                        table = pd.read_csv("sakila-db/sakila-csv/" + fName)
                    table.columns = cleanColNames(table)
                    print(fName)
                    sqlName = fName.split(".")[0]
                    tables[sqlName] = table
        
        # Prints info on the current tables
        elif command == "tableInfo":
            attrib_total = 0
            row_total = 0
            for fName in tables:
                table = tables[fName]
                print(fName)
                print(table.shape)
                row_total += table.shape[0]
                attrib_total += table.shape[1]
                densities = []
                for attrib in table.columns:
                    attrib_density = len(set(table[attrib]))/table.shape[0]
                    densities.append(attrib_density)
            print("Total rows: " + str(row_total))
            print("Total attribs: " + str(attrib_total))
        
        # Generates pickle files with non empty joins
        elif command == "getNonEmpty":
            connections_dict = defaultdict(list)
            
            for iFile in range(len(sak_file_names)):
                file1 = sak_file_names[iFile].split(".")[0]
                table1 = tables[file1]
                for attrib1 in table1.columns:
                    for jFile in range(iFile + 1, len(sak_file_names)):
                        file2 = sak_file_names[jFile].split(".")[0]
                        table2 = tables[file2]
                        for attrib2 in table2.columns:
                            non_empty = getJoinNonEmpty(file1, table1, attrib1, file2, table2, attrib2)
                            if non_empty:
                                connections_dict[file1 + "." + attrib1].append(file2 + "." + attrib2)
            
            pickle.dump(connections_dict, open("sakila_joins.p", "wb" ) )
            connections_dict = defaultdict(list)
            
            chi_file_names = os.listdir('chicago15')
            chi_file_names = list(filter(lambda x : not x == '.DS_Store', chi_file_names))
            tables = {}
            for fName in chi_file_names:
                if not fName == ".DS_Store":
                    table = pd.read_csv("chicago15/" + fName)
                    table.columns = cleanColNames(table)
                    print(fName)
                    sqlName = fName.split(".")[0]
                    tables[sqlName] = table
            for iFile in range(len(chi_file_names)):
                file1 = chi_file_names[iFile].split(".")[0]
                table1 = tables[file1]
                for attrib1 in table1.columns:
                    for jFile in range(iFile + 1, len(chi_file_names)):
                        file2 = chi_file_names[jFile].split(".")[0]
                        table2 = tables[file2]
                        for attrib2 in table2.columns:
                            non_empty = getJoinNonEmpty(file1, table1, attrib1, file2, table2, attrib2)
                            if non_empty:
                                connections_dict[file1 + "." + attrib1].append(file2 + "." + attrib2)
            pickle.dump(connections_dict, open("chicago_joins.p", "wb" ) )
        
        # Runs single attribute join experiments
        elif command == "singleJoins":
            join_types = defaultdict(int)
            
            for iFile in range(len(sak_file_names)):
                file1 = sak_file_names[iFile].split(".")[0]
                table1 = tables[file1]
                for attrib1 in table1.columns:
                    for jFile in range(iFile + 1, len(sak_file_names)):
                        file2 = sak_file_names[jFile].split(".")[0]
                        table2 = tables[file2]
                        for attrib2 in table2.columns:
                            join_type = getJoinType(file1, table1, attrib1, file2, table2, attrib2)
                            print("-------------  Join is " + join_type)
                            join_types[join_type] += 1
            for iFile in range(len(chi_file_names)):
                file1 = chi_file_names[iFile].split(".")[0]
                table1 = tables[file1]
                for attrib1 in table1.columns:
                    for jFile in range(iFile + 1, len(chi_file_names)):
                        file2 = chi_file_names[jFile].split(".")[0]
                        table2 = tables[file2]
                        for attrib2 in table2.columns:
                            join_type = getJoinType(file1, table1, attrib1, file2, table2, attrib2)
                            print("-------------  Join is " + join_type)
                            join_types[join_type] += 1
                            
            # Defunct. Code used to graph access percentage vs bandwidth ratios
            # -------------------------------------------------------------
            # # Graph
            # percentages_11 = []
            # percentages_1M = []
            # percentages_MM = []
            # aff_11_bandwidth_ratios = []
            # aff_1M_bandwidth_ratios = []
            # aff_MM_bandwidth_ratios = []
            # fp_11_bandwidth_ratios = []
            # fp_1M_bandwidth_ratios = []
            # fp_MM_bandwidth_ratios = []
            # pp_11_bandwidth_ratios = []
            # pp_1M_bandwidth_ratios = []
            # pp_MM_bandwidth_ratios = []
            # for i in range(len(access_percentages)):
            #     if categories[i] == "1-1":
            #         percentages_11.append(access_percentages[i])
            #         aff_11_bandwidth_ratios.append(aff_bandwidth_ratios[i])
            #         fp_11_bandwidth_ratios.append(fp_bandwidth_ratios[i])
            #         pp_11_bandwidth_ratios.append(pp_bandwidth_ratios[i])
            #     if categories[i] == "1-Many":
            #         percentages_1M.append(access_percentages[i])
            #         aff_1M_bandwidth_ratios.append(aff_bandwidth_ratios[i])
            #         fp_1M_bandwidth_ratios.append(fp_bandwidth_ratios[i])
            #         pp_1M_bandwidth_ratios.append(pp_bandwidth_ratios[i])
            #     if categories[i] == "Many-Many":
            #         percentages_MM.append(access_percentages[i])
            #         aff_MM_bandwidth_ratios.append(aff_bandwidth_ratios[i])
            #         fp_MM_bandwidth_ratios.append(fp_bandwidth_ratios[i])
            #         pp_MM_bandwidth_ratios.append(pp_bandwidth_ratios[i])
            # 
            # plt.yscale("log")
            # plt.plot(percentages_11, aff_11_bandwidth_ratios, 'r.')
            # plt.plot(percentages_11, fp_11_bandwidth_ratios, 'g.')
            # plt.plot(percentages_11, pp_11_bandwidth_ratios, 'b.')
            # # plt.plot(percentages_1M, aff_1M_bandwidth_ratios, 'r^')
            # # plt.plot(percentages_1M, fp_1M_bandwidth_ratios, 'g^')
            # # plt.plot(percentages_1M, pp_1M_bandwidth_ratios, 'b^')
            # # plt.plot(percentages_MM, aff_MM_bandwidth_ratios, 'rs')
            # # plt.plot(percentages_MM, fp_MM_bandwidth_ratios, 'gs')
            # # plt.plot(percentages_MM, pp_MM_bandwidth_ratios, 'bs')
            # # plt.plot(access_percentages, pp_bandwidth_ratios, 'r.', access_percentages, fp_bandwidth_ratios, 'b.', access_percentages, aff_bandwidth_ratios, 'g.')
            # plt.show()
            # -------------------------------------------------------------
                
            for join_key, join_val in join_types.items():
                pp_to_fp_mean_bandwidth[join_key] = pp_to_fp_mean_bandwidth[join_key] / join_val
                pp_to_fp_mean_leakage[join_key] = pp_to_fp_mean_leakage[join_key] / join_val
                fp_to_aff_mean_leakage[join_key] = fp_to_aff_mean_leakage[join_key] / join_val
            
                print(join_key + " - " + str(join_val))
                print("Min/Mean/Max PP to FP b: " + str(pp_to_fp_min_bandwidth[join_key]) + "/" + str(pp_to_fp_mean_bandwidth[join_key]) + "/" + str(pp_to_fp_max_bandwidth[join_key]))
                print("Min/Mean/Max PP to FP l: " + str(pp_to_fp_min_leakage[join_key]) + "/" + str(pp_to_fp_mean_leakage[join_key]) + "/" + str(pp_to_fp_max_leakage[join_key]))
                print("Min/Mean/Max FP to AFF l: " + str(fp_to_aff_min_leakage[join_key]) + "/" + str(fp_to_aff_mean_leakage[join_key]) + "/" + str(fp_to_aff_max_leakage[join_key]))
        
        else:
            print("Command not recognized. Try \"help\" to see the list of test harness commands")
    

# Runs the main function
if __name__== "__main__":
    main()