import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd
import os
import cv2
import csv
import math
import open3d as o3d
import numpy as np
import plotly.graph_objects as go
import dash
from dash import dcc
from dash import html
from tkhtmlview import HTMLLabel
import shutil as sop
import copy
import triangle as tr
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd



class ZoomInZoomOutFunctionality():
    # This class is developed for providing the zoom-in and zoom-out functionality when we use openCV module
    # Apart from last 'reconstructWindow' method we can use this idea for developing other module's zoom-in 
    # & zoom-out if any module provide only static image view functionality
    # Here, two scale factor used for x and y direction so that in case of extra exploration in one dimension 
    # other should not get affected i.e. provide relaxation where one dimension require zoom but not other 

    def redefineScale(self):
        # This will ensure the proper scale factor value to be noted before proceeding to next operation
        self.scaleFactorX = (float(self.imageFinalX-self.imageInitialX))/(self.resizedDimensions[0])
        self.scaleFactorY = (float(self.imageFinalY-self.imageInitialY))/(self.resizedDimensions[1])


    def zoomInNextCoordinates(self,x,y):
        # Find next coordinates when zoom-in is called, here default incremental zoom percentage is hard-coded 
        # as 10% But by using other data-structure we can use a steping zoom-percentage as we can observe in the 
        # photo viewer application (Where there is steping values set example 100% -> 110% -> 121% -> 133% ...)
        # Fact used here: Linear interpolation of 10% length from mouse pointer to all four directions
        self.imageInitialX = self.imageInitialX + int((float(x))*(1-1/1.1)*self.scaleFactorX)
        self.imageFinalX = self.imageFinalX - int((float(self.resizedDimensions[0]-x))*(1-1/1.1)*self.scaleFactorX)
        self.imageInitialY = self.imageInitialY + int((float(y))*(1-1/1.1)*self.scaleFactorY)
        self.imageFinalY = self.imageFinalY - int((float(self.resizedDimensions[1]-y))*(1-1/1.1)*self.scaleFactorY)


    def zoomOutNextCoordinates(self,x,y):
        # Same description applicable as in zoom-in and additionally we have to handle error handling when new 
        # coordinates after extrapolation goes outside of the imagepixels
        # Below code is for getting extrapolated coordinates, also here extrapolation is 10% (hard-coded) of original value
        self.imageInitialX = self.imageInitialX - int((float(x))*0.1*self.scaleFactorX)
        self.imageFinalX = self.imageFinalX + int((float(self.resizedDimensions[0]-x))*0.1*self.scaleFactorX)
        self.imageInitialY = self.imageInitialY - int((float(y))*0.1*self.scaleFactorY)
        self.imageFinalY = self.imageFinalY + int((float(self.resizedDimensions[1]-y))*0.1*self.scaleFactorY)

        # Handle the corner case where increment/decrement is behond the pixels of image
        self.imageInitialX = max(self.imageInitialX, 0)
        self.imageFinalX = min(self.imageFinalX, self.originalImageWidth)
        self.imageInitialY = max(self.imageInitialY, 0)
        self.imageFinalY = min(self.imageFinalY, self.originalImageHeight)


    def setImageWindowDimensions(self, windowWidth=950, windowHeight=650):
        # Window size in which we want to show image
        # windowWidth and windowHeight can be custom set if not provided, then image would be 950x650
        # Used fact: Dimension of image which is exceeding largest from our desired window size will be 
        # reduced to our desired size window and remaining dimension will be adjusted on that same scale
        
        # Below two are top-left corner coordinates which will help in cropping from original image and initialising them
        self.imageInitialX = 0
        self.imageInitialY = 0

        # Below two are bottom-right corner coordinates which will help in cropping from original image and initialising them
        self.imageFinalX = self.newImage.shape[1]
        self.imageFinalY = self.newImage.shape[0]

        # Variables to store original image dimesion values
        self.originalImageWidth = self.newImage.shape[1]
        self.originalImageHeight = self.newImage.shape[0]

        # Determine which dimension is exceeding largest and then set scale values accordingly
        if(self.originalImageHeight-windowHeight < self.originalImageWidth-windowWidth):
            self.scaleFactorX, self.scaleFactorY = (float(self.originalImageWidth))/windowWidth, (float(self.originalImageWidth))/windowWidth
        else:
            self.scaleFactorX, self.scaleFactorY = (float(self.originalImageHeight))/windowHeight, (float(self.originalImageHeight))/windowHeight

        # As per selected above scale values, set window dimesions
        self.resizedDimensions = (int(self.originalImageWidth/self.scaleFactorX),int(self.originalImageHeight/self.scaleFactorY))


    def zoomInZoomOut(self, x, y, flags):
        # Landing method for determine next coordinate from the above methods by appropriate action called from parent method
        
        if (flags>0):   # This is for zoom-in
            # For accuracy purpose, scale is defined again
            self.redefineScale()
            # Make sure that it is zoomable only upto a limited scale value
            if(self.scaleFactorX<0.3 or self.scaleFactorY<0.3):
                return
            # If passed from above condition then set new coordinates
            self.zoomInNextCoordinates(x,y)

        else:   # This is for zoom-out
            # For accuracy purpose, scale is defined again
            self.redefineScale()
            # Make sure that it is zoom-out only upto a limited scale value
            if(self.scaleFactorX>10 or self.scaleFactorY>10):
                return
            # If passed from above condition then set new coordinates
            self.zoomOutNextCoordinates(x,y)


    def reconstructWindow(self):
        # This is specific to OpenCV, After re-coordinating, show updated window for live effects
        # Read image from the specific location
        self.newImage = cv2.imread(self.imageLocation, 1)
        # Crop specific part of that image to show
        self.newImage = self.newImage[self.imageInitialY:self.imageFinalY, self.imageInitialX:self.imageFinalX]
        # Display image on the opneCV window
        self.newImage = cv2.resize(self.newImage, self.resizedDimensions)



class ProcessSurfaceFlowAndFinalGraphOutput(tk.Tk):

    def __init__(self, fileLocation, outletNode):
        
        # Initialise the wideget window propertiess and neccessary variables
        super().__init__()
        self.title('Outflow Hydrograph...')
        self.geometry('850x400')
        self.minsize(850,420)
        self.maxsize(850,420)
        self.fileLocation = fileLocation
        self.outletNode = outletNode
        # Get current working directory so that background image could be find out
        currentWorkingDirectory = os.getcwd()
        imageFileLocation = currentWorkingDirectory + '\i1.png'
        pathToIcon = currentWorkingDirectory + r'\guiIcon.ico'
        # Set icon for the widget window
        self.wm_iconbitmap(pathToIcon)
        self.backgroundImageObject = tk.PhotoImage(file = imageFileLocation)
        # set background image with the help of canvas
        self.newCanvas = tk.Canvas(self)
        self.newCanvas.create_image(0,0,anchor=tk.NW,image=self.backgroundImageObject)
        self.newCanvas.place(x=0,y=0,relheight=1,relwidth=1)

        # Start simulation computation after decide of the final geometry of the catchment for output
        self.startCalculating()

        # Labels for maximum flow rate and the corresponding time when it occured, will be shown on right side of the flood hydrograph
        ttk.Label(self, text='Maximum flow rate obtained:', foreground='red', font='Times 17 italic bold').place(x=520,y=100,anchor=tk.W)
        ttk.Label(self, text='And it occured at time, t:', foreground='red', font='Times 17 italic bold').place(x=520,y=250,anchor=tk.W)


    def startCalculating(self):
        ################### CODE STARTS HERE: Finding flow over connected triangular cells and final drop into valley segments ###################
        
        # The code computes surface flow over triangular cells by a zero-inertia model
        # Flow exchange between cells across a common boundary is approximated by Manning's equation

        manCoeff = 0.01
        deltaTime = 10
        totalSimulationTime = 7000
        rainFallRate = 0.01/3600    # 1 cm rainfall occuring for 1 hour period
        initialWaterDepth = rainFallRate*deltaTime      # it's better to start with some initial water depth

        # Below are for storing the inidividual traingular cell properties
        trianglesNodes = []
        triangles = []
        trianglesArea = []
        trianglesCG = []
        trianglesAverageGroundElevation = []
        trianglesWaterLevel = []

        ####################### CODE STARTS HERE: Defining the triagular cells from previously provided geometrical data #########################
        
        # Get all triangular/Non-triangular nodes from provided geometry
        newFileName = self.fileLocation + '/finalNodes.txt'
        pointCloud = np.loadtxt(newFileName,skiprows=0,ndmin=2)
        xyz = pointCloud[:,:]
        xyFormat = []   # Separate x&y coordinates so that 2D triangulation could be carried out
        
        for ele in xyz:
            xyFormat.append([ele[0], ele[1]])
            trianglesNodes.append([ele[0], ele[1], ele[2]])

        # List all the segments connecting nodes that must be preset in the final triangulation
        newFileName = self.fileLocation + '/finalSegments.txt'
        pointCloud = np.loadtxt(newFileName,skiprows=0,ndmin=2)
        segs = pointCloud[:,:]
        mandatorySegments = []

        for ele in segs:
            mandatorySegments.append((int(ele[0]), int(ele[1])))

        # Carry out triangulation part: here 'pA' will do triangulation for those nodes which are present inside the 
        # closed boundary, hence catchment boundary segments plays here great roles, all those nodes which were selected 
        # outside the catchment boundary now will be no longer part of triangles, but since intersection points are 
        # already done so there is no error in formation of traingles along with ridge line or valley lines And also 
        # no new nodes will be created automatically
        triangulatedData = tr.triangulate({'vertices':xyFormat, 'segments':mandatorySegments}, 'pA')
        tG = triangulatedData['triangles']  # This will give 2D list of the all nodes of triangles

        # store triangle's nodes in sorted order so that we can use this property for later use
        for ele in tG:
            nodeList = [ele[0], ele[1], ele[2]]
            nodeList.sort()
            triangles.append([nodeList[0], nodeList[1], nodeList[2]])
        
        # Create a neighbour triangle track
        # Start three indices have indicator number which represent as follows: 
        # 1 -> There exist a neighbour traingle, 0 -> No flow boundry, -1 -> water transfer to valley segment
        # Next three indices will point the respective triangle or valley segment and -1 means no data available (i.e. no flow boundry) 
        # And 'trianglesEdgeLengths' is 2D list where 0th index will store edge length between first and second node and similarly for 1st and 2nd index.
        # And 'neighbourTriangleCGDistance' is distance between the CG of current triangle and its neighbouring triangles in order of edge assumed
        neighbourTriangle = [[0, 0, 0, -1, -1, -1] for i in range(len(triangles))]
        trianglesEdgeLengths = [[0, 0, 0] for i in range(len(triangles))]
        neighbourTriangleCGDistance = [[-1, -1, -1] for i in range(len(triangles))]

        # Key note: We are assuming triangular cells are in horizontal plane hence all distances are in that way and it height from the datum point 
        # is average of z-coordinate of the its nodes

        for i in range(len(triangles)):     #Iterate over all triangles
            # Store the x,y,z coordinates of nodes of triangle
            x1, y1, z1 = trianglesNodes[triangles[i][0]][0], trianglesNodes[triangles[i][0]][1], trianglesNodes[triangles[i][0]][2]
            x2, y2, z2 = trianglesNodes[triangles[i][1]][0], trianglesNodes[triangles[i][1]][1], trianglesNodes[triangles[i][1]][2]
            x3, y3, z3 = trianglesNodes[triangles[i][2]][0], trianglesNodes[triangles[i][2]][1], trianglesNodes[triangles[i][2]][2]

            area = abs(0.5*(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2)))      # Calculate area of triangle from the shoelace formula
            trianglesArea.append(area)
            trianglesCG.append([(x1+x2+x3)/3, (y1+y2+y3)/3])
            trianglesAverageGroundElevation.append((z1+z2+z3)/3)
            trianglesWaterLevel.append(trianglesAverageGroundElevation[i]+initialWaterDepth)       # Water level is wrt datum point

            for j in range(3):      # iterate over its edges one by one
                x1, y1 = trianglesNodes[triangles[i][j]][0], trianglesNodes[triangles[i][j]][1]
                x2, y2 = trianglesNodes[triangles[i][(j+1)%3]][0], trianglesNodes[triangles[i][(j+1)%3]][1]

                trianglesEdgeLengths[i][j] = math.sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1))

                for k in range(len(triangles)):     # iterate over all the triangles for finding common edge between above iterated triangle and remainings
                    if i==k:    # we are on same triangle so have to escape
                        continue

                    for l in range(3):  # iterate over all the edges of the triangle and compare with our main triangle to find if there is any common edge present
                        # There is no problem for the first two indices but since third edge is between node 3 and 1 so there is just reverse hence have to 
                        # separately consider the last edge where we compare 1st node first and 3rd node second
                        if ((j<2) and (l<2) and (triangles[i][j] == triangles[k][l]) and (triangles[i][(j+1)%3] == triangles[k][(l+1)%3])):
                            pass
                        elif ((j==2) and (l<2) and (triangles[i][0] == triangles[k][l]) and (triangles[i][2] == triangles[k][(l+1)%3])):
                            pass
                        elif ((l==2) and (j<2) and (triangles[i][j] == triangles[k][0]) and (triangles[i][(j+1)%3] == triangles[k][2])):
                            pass
                        elif ((l==2) and (j==2) and (triangles[i][0] == triangles[k][0]) and (triangles[i][2] == triangles[k][2])):
                            pass
                        else:
                            continue

                        # If we found common edge, note down that triangle's serial number
                        neighbourTriangle[i][j] = 1
                        neighbourTriangle[i][j+3] = k

        # Below are no flow segments which may be catchment boundary segments or ridge segments, we note down these so that in case we find any 
        # of this segment we should make proper arrangment such that water from that triangular cell doesn't shift to neighbour cell
        newFileName = self.fileLocation + '/noFlowSegments.txt'
        pointCloud = np.loadtxt(newFileName,skiprows=0,ndmin=2)
        seg = pointCloud[:,:]
        noFlowSegments = []

        for ele in seg:
            nodeList = [int(ele[0]), int(ele[1])]
            nodeList.sort()
            noFlowSegments.append([nodeList[0], nodeList[1]])

        for i in range(len(triangles)):     # iterate over all triangles and mark segments through which there would be no flow
            for j in range(3):      # iterate over all three edges
                for k in range(len(noFlowSegments)):    # iterate over all no flow segments
                    if ((j<2) and (triangles[i][j] == noFlowSegments[k][0]) and (triangles[i][(j+1)%3] == noFlowSegments[k][1])):
                        pass
                    elif ((j==2) and (triangles[i][0] == noFlowSegments[k][0]) and (triangles[i][2] == noFlowSegments[k][1])):
                        pass
                    else:
                        continue
                    
                    # Marking that edge as no flow boundary, and assigning -1 value for neighbouring triangle which indicate 
                    # there is no neighbouring cell
                    neighbourTriangle[i][j] = 0
                    neighbourTriangle[i][j+3] = -1

        # Take note of all full transfer segments i.e. valley segments
        newFileName = self.fileLocation + '/fullTransferSegments.txt'
        pointCloud = np.loadtxt(newFileName,skiprows=0,ndmin=2)
        seg = pointCloud[:,:]
        valleySegments = []

        for ele in seg:
            nodeList = [int(ele[0]), int(ele[1])]
            nodeList.sort()
            valleySegments.append([nodeList[0], nodeList[1]])

        # Below array is for valley segment's length which will be used during application of manning equation where length of flow segemnt will 
        # be required and the Assigning hypothetical value of 1e-1 is important because there would be some segemnts which lies outside of catchment 
        # boundary hence they are not part of triangles but since they are there in valley segments list we have to account them
        valleySegmentsLength = [1e-1]*len(valleySegments) 

        for i in range(len(triangles)):     # Iterate over all triangles
            for j in range(3):      # Iterate over all edges
                for k in range(len(valleySegments)):    # Iterate over all valley segments
                    if ((j<2) and (triangles[i][j] == valleySegments[k][0]) and (triangles[i][(j+1)%3] == valleySegments[k][1])):
                        pass
                    elif ((j==2) and (triangles[i][0] == valleySegments[k][0]) and (triangles[i][2] == valleySegments[k][1])):
                        pass
                    else:
                        continue

                    # Found valley segment mark it
                    neighbourTriangle[i][j] = -1
                    neighbourTriangle[i][j+3] = k   # Ener index number of this valley segment
                    valleySegmentsLength[k] = trianglesEdgeLengths[i][j]    # Triangle edge is that valley segment

        # Below iteration is for getting center to center distance between the neighbouring triangles  
        for i in range(len(triangles)):
            for j in range(3):
                if neighbourTriangle[i][j] == 1:    # Note here taht we are not accounting when we encounter a full transfer segment i.e. a valley segemnt
                    cgx1, cgy1 = trianglesCG[i][0], trianglesCG[i][1]
                    cgx2, cgy2 = trianglesCG[neighbourTriangle[i][j+3]][0], trianglesCG[neighbourTriangle[i][j+3]][1]
                    neighbourTriangleCGDistance[i][j] = math.sqrt((cgx2-cgx1)*(cgx2-cgx1) + (cgy2-cgy1)*(cgy2-cgy1))

        ####################### CODE ENDS HERE: Defining the triagular cells from previously provided geometrical data ############################

        totalIntervals = int(totalSimulationTime/deltaTime) + 10
        # Idea used here is that simulate the surface flow over the triangular cells and store value of water amount that is dumped into the valley 
        # segments after each time interval
        valleySegmentsCollectionOnIntervalTime = [[0.0]*len(valleySegments) for i in range(totalIntervals)]

        for i in range(deltaTime, totalSimulationTime+1, deltaTime):    # Iterate over all interval of time
            # Matrices that required for computation
            matA = [[0.0]*len(triangles) for i in range(len(triangles))]
            matB = [0.0]*len(triangles)

            # First target is to develop the unit hydrograph of the catchment. Coming versions can include the rainfall rate as per the provided 
            # from the user inputs and also would not be only limited to the 3600 seconds but user inputed
            if i>3600:
                rainFallRate = 0

            for j in range(len(triangles)):     # Iterate over all triangles
                matA[j][j] = trianglesArea[j]/deltaTime
                matB[j] = matB[j] + rainFallRate*trianglesArea[j]

                for k in range(3):      # Iterate over all edges

                    if neighbourTriangle[j][k] == 1:    # When there is neighbouring triangle for that edge is present
                        # General variables for coming calculations
                        maxGroundElevation = max(trianglesAverageGroundElevation[j], trianglesAverageGroundElevation[neighbourTriangle[j][k+3]])
                        edgeLength = trianglesEdgeLengths[j][k]
                        cgCenterToCgCenterDistance = neighbourTriangleCGDistance[j][k]
                        waterLevelDifference = trianglesWaterLevel[neighbourTriangle[j][k+3]] - trianglesWaterLevel[j]
                        
                        if waterLevelDifference > 1e-9:     # We should only consider the water transfer when there is significant water level difference
                            heightOfFlow = trianglesWaterLevel[neighbourTriangle[j][k+3]] - maxGroundElevation
                            # Below are the governing manning equations which are used for the computation
                            Qik = edgeLength/(manCoeff*math.sqrt(cgCenterToCgCenterDistance))*(heightOfFlow**1.67)*math.sqrt(waterLevelDifference)
                            dQik_dzi = -edgeLength/(manCoeff*math.sqrt(cgCenterToCgCenterDistance))*((heightOfFlow**1.67)*0.5/math.sqrt(waterLevelDifference))
                            dQik_dzk = edgeLength/(manCoeff*math.sqrt(cgCenterToCgCenterDistance))*(1.67*(heightOfFlow**0.67)*math.sqrt(waterLevelDifference) + (heightOfFlow**1.67)*0.5/math.sqrt(waterLevelDifference))

                            # Enter values in the matrices
                            matA[j][j] = matA[j][j] - dQik_dzi
                            matA[neighbourTriangle[j][k+3]][j] = matA[neighbourTriangle[j][k+3]][j] + dQik_dzi
                            matA[j][neighbourTriangle[j][k+3]] = matA[j][neighbourTriangle[j][k+3]] - dQik_dzk
                            matA[neighbourTriangle[j][k+3]][neighbourTriangle[j][k+3]] = matA[neighbourTriangle[j][k+3]][neighbourTriangle[j][k+3]] + dQik_dzk

                            matB[j] = matB[j] + Qik
                            matB[neighbourTriangle[j][k+3]] = matB[neighbourTriangle[j][k+3]] - Qik
                    
                    elif neighbourTriangle[j][k] == -1:     # When edge is a valley segemnt. Hence, amount to be transfered into the valley segment
                        heightOfFlow = trianglesWaterLevel[j] - trianglesAverageGroundElevation[j]
                        edgeLength = trianglesEdgeLengths[j][k]
                        # Below are the governing weir equations which are used for the computation
                        Qik = -0.6*1.706*edgeLength*(heightOfFlow**1.5)
                        dQik_dzi = -0.6*1.706*edgeLength*1.5*math.sqrt(heightOfFlow)
                        # Enter values in the matrices
                        matA[j][j] = matA[j][j] - dQik_dzi
                        matB[j] = matB[j] + Qik
                        # Qik amount of water is transfered into the valley segment
                        valleySegmentsCollectionOnIntervalTime[int(i/deltaTime)][neighbourTriangle[j][k+3]] = valleySegmentsCollectionOnIntervalTime[int(i/deltaTime)][neighbourTriangle[j][k+3]] - Qik
            
            # Solve matrices to find the delta values
            try:
                mat1 = np.array(matA)
                mat2 = np.array(matB)
                deltaDifference = np.linalg.solve(mat1, mat2)
            except:
                # In case matrix is singular by some reasons we may escape soltion and write down delta values as zero
                deltaDifference = [0.0]*len(triangles)

            for j in range(len(deltaDifference)):   # Iterate over all triangles and adjust the water levels accordingly
                trianglesWaterLevel[j] = trianglesWaterLevel[j] + deltaDifference[j]
                if trianglesWaterLevel[j] < trianglesAverageGroundElevation[j] + rainFallRate*deltaTime:    # There should be minimum water depth present in the cell
                    trianglesWaterLevel[j] = trianglesAverageGroundElevation[j] + rainFallRate*deltaTime

        ##################### CODE ENDS HERE: Finding flow over connected triangular cells and final drop into valley segments #####################

        ################## CODE STARTS HERE: Find the valley segment's order to make arrangement for flow through valley segments ##################

        
        backNodeToSegment = {}      # This dictionary will help in find out the index number of valley segments corresponding to a valley node

        for i in range(len(valleySegments)):    # Iterate over all valley segments
            # Here dealing with first/start node of segment
            if valleySegments[i][0] in backNodeToSegment:   # If node is already there in dictionary
                backNodeToSegment[valleySegments[i][0]].append(i)
            else:
                backNodeToSegment[valleySegments[i][0]] = []    # Otherwise just initialise list
                backNodeToSegment[valleySegments[i][0]].append(i)   # And insert index number of that segment

            # Here dealing with second/end node of segment
            if valleySegments[i][1] in backNodeToSegment:   # If node is already there in dictionary
                backNodeToSegment[valleySegments[i][1]].append(i)
            else:
                backNodeToSegment[valleySegments[i][1]] = []    # Otherwise just initialise list
                backNodeToSegment[valleySegments[i][1]].append(i)   # And insert index number of that segment
        
        # In order to remove duplicates and maintain uniqueness, all nodes which are processed will be stored in below list
        QueueForProcessing = [self.outletNode]      # Initialise the list with outlet node, this is starting point for process
        topologicalListToSolve = []     # This will store topological storage of the segments i.e. from downstream to upstream order of segments

        for node1 in QueueForProcessing:    # Iterate in the order of the inserted nodes for processing
            for seg1 in backNodeToSegment[node1]:   # Iterate over all the segments that are connected to the node
                # Here Idea used: Find the other node of the segment of interest, if node is upstaream then only useful (otherwise water will not 
                # flow from that segment in over desired direction but in reverse direction which is of no use, this fact is check from comparing 
                # elevation of nodes) and also node must not be processed previously otherwise it will fall in loop where it will not resolved
                # Since segment end nodes are sorted hence we have to check on both end nodes, so put nodes in if-else statement
                 
                if valleySegments[seg1][0] != node1 and trianglesNodes[valleySegments[seg1][0]][2] >= trianglesNodes[node1][2] and (valleySegments[seg1][0] not in QueueForProcessing):
                    QueueForProcessing.append(valleySegments[seg1][0])      # If above condition already met we should add it to list where it will get resolved in next iterations
                    topologicalListToSolve.append([valleySegments[seg1][0], seg1, node1])   # we are storing in order of [downstream node, segment index, upstream node]

                elif valleySegments[seg1][1] != node1 and trianglesNodes[valleySegments[seg1][1]][2] >= trianglesNodes[node1][2] and (valleySegments[seg1][1] not in QueueForProcessing):
                    QueueForProcessing.append(valleySegments[seg1][1])      # Same as above
                    topologicalListToSolve.append([valleySegments[seg1][1], seg1, node1])   # Same as above

        # We require topological sorting from upstream to downstream so that water transfer could be made easily by following these segments order
        # and will require just transfer from one node to another node in order. But Since list made is from downstream to upsrtream, we have to reverse it
        topologicalListToSolve.reverse()

        # For the efficiency in the matrix storage of node's values, we can assign number from zero onwards to the actual node numbering
        nodeToNormalNumbering = {}
        cnt = 0
        for nodeNumber in backNodeToSegment:
            nodeToNormalNumbering[nodeNumber] = cnt
            cnt = cnt + 1

        ################### CODE ENDS HERE: Find the valley segment's order to make arrangement for flow through valley segments ###################

        ################### CODE STARTS HERE: Transfer of water from one valley node to another using Kinetic wave approximation ###################

        b0 = 25     # Width of rectangular channel
        mn = 0.035  # Manning's roughness coefficient for channel
        partitionOnLength = 10  # Total partitions considered on channel length of one segment
        partitionOnTime = 10    # Total partitions considered on the deltatime for one segment
        dt = (float(deltaTime))/partitionOnTime # New time incremental value
        beeta = 0.6
        b1 = beeta - 1

        # By using this method, we will require to have a memory complexity of O(partitionOnTime*partitionOnLength*NoOfValleySegments*TotalNoOfSimulationIntervals) 
        # and same goes for the time complexity. However we can't optimise on time from this method but we can certainly do for memory complexity. 
        # Here, matB is useful for computation of one segment at a time for one simulation interval after that these results will be saved in matA1 and matA2. 
        # matA1 is initialised for all valley nodes And matA2 is initialised for all valley segments
        matA1 = [[1e-6]*partitionOnTime for i in range(len(nodeToNormalNumbering))]
        matA2 = [[1e-6]*partitionOnLength for i in range(len(valleySegments))]
        matB = [[0.0]*partitionOnLength for i in range(partitionOnTime)]
        finalXYGraphPoints = [[],[]]    # Here we store outlet node's outflow value with time stamp

        for i in range(deltaTime, totalSimulationTime, deltaTime):      # Iterate over all simulation intervals
            # n1 - upstream valley node, n2 - downstream valley node, s1 - valley segment, lastDischarge - outletnode's outflow value track
            n1, s1, n2, lastDischarge = 0, 0, 0, 0

            for tl in topologicalListToSolve:   # Iterate over all topological sorted valley segments

                for j in range(len(matB)):
                    for k in range(len(matB[0])):
                        matB[j][k] = 0.0        # Intialise matB with zeros for next computation

                n1, s1, n2 = tl[0], tl[1], tl[2]    # Assigning values

                dx = valleySegmentsLength[s1]/(float(partitionOnLength-1))      # delta value for the channel length
                    
                if dx < 1e-7:   # If delta value is too small we can neglect remaining calculations
                    for j in range(1, partitionOnTime):
                        matA1[nodeToNormalNumbering[n2]][j] = matA1[nodeToNormalNumbering[n2]][j] + matA1[nodeToNormalNumbering[n1]][j] + valleySegmentsCollectionOnIntervalTime[int(i/deltaTime)][s1]/(float(partitionOnTime))
                        matA1[nodeToNormalNumbering[n1]][j] = 0
                    continue
                
                s0 = (trianglesNodes[n1][2]-trianglesNodes[n2][2])/valleySegmentsLength[s1]     # Slope of the channel bed
                if s0<0.01:     # There should be minimum channel bed slope so that water could be transfered
                    s0 = 0.01

                alpha = (mn*(b0**0.667)/math.sqrt(s0))**0.6
                
                dtdx = dt/dx

                for j in range(1, partitionOnLength):
                    matB[0][j] = matA2[s1][j]       # Initialise the matB, these values signifies the last computation carried out before this simulation time interval

                for j in range(1, partitionOnTime):
                    matB[j][0] = matA1[nodeToNormalNumbering[n1]][j]    # Similarly, we intialise from the upstream node of the segment for current simulation time interval

                # Below is kinematic wave approximation for current simulation interval of our desired valley segment
                for j in range(1, partitionOnTime):
                    for k in range(1, partitionOnLength):
                        # Below are governing equations for the kinematic wave approximation
                        Qavg = 0.5*(matB[j-1][k] + matB[j][k-1])
                        numo = dtdx*matB[j][k-1] + alpha*beeta*matB[j-1][k]*(Qavg**b1) + dt*(valleySegmentsCollectionOnIntervalTime[int(i/deltaTime)][s1]/(deltaTime*partitionOnLength))
                        deno = dtdx + (alpha*beeta*(Qavg**b1))
                        matB[j][k] = numo/deno

                for j in range(1, partitionOnTime):
                    matA1[nodeToNormalNumbering[n1]][j] = 0     # Initialise upstream node which will be used in next simulation time interval
                    matA1[nodeToNormalNumbering[n2]][j] = matA1[nodeToNormalNumbering[n2]][j] + matB[j][partitionOnLength-1]    # Update downstream node

                for j in range(1, partitionOnLength):
                    matA2[s1][j] = matB[partitionOnTime-1][j]   # Save computation to resume in next simulation time interval
                
            for j in range(1, partitionOnTime):
                if j+1 == partitionOnTime:
                    lastDischarge = matA1[nodeToNormalNumbering[n2]][j]     # Last value would be our outlet node's outflow value at the end of simulation time interval
                matA1[nodeToNormalNumbering[n2]][j] = 0.0   # Initialise outlet node with zero for next simulation time interval

            # Save discharge value against time value
            finalXYGraphPoints[0].append(i)
            finalXYGraphPoints[1].append(lastDischarge)

        #################### CODE ENDS HERE: Transfer of water from one valley node to another using Kinetic wave approximation ####################

        ########################## CODE STARTS HERE: Another method for transfer of water from one valley node to another ##########################
        
        # Idea used here: Valley segments are of uniform length and time require for full transfer of the water from a segment is equal to the one 
        # simulation time interval. This is less accurate but can be used for verification purpose where we can find a peak discharge value

        valleyIndicatorNumber = [0]*len(valleySegments)     # This number will indicate how far is our segment from the outlet node (in term of, number of middle segments)
        listOfStarter = []      # List for appending valley nodes to be process
        listOfStarter.append([self.outletNode, 1])      # Initialising with outlet node and distance is one unit

        for node1 in listOfStarter:     # Iterating on the nodes to be processed
            for seg1 in backNodeToSegment[node1[0]]:    # Iterate on all segments which are connected to our processing node
                if valleyIndicatorNumber[seg1] == 0:    # Only process segment if it didn't get processed yet
                    valleyIndicatorNumber[seg1] = node1[1]  # Assign priority number
                    if valleySegments[seg1][0] == node1[0]: # If this node is under process then append other node with next priority number
                        listOfStarter.append([valleySegments[seg1][1], node1[1]+1])
                    else:   # Similarly here
                        listOfStarter.append([valleySegments[seg1][0], node1[1]+1])
        
        totalIntervals = totalIntervals + len(valleySegments)       # Have to provide extra intervals so that most upstream water will be discharged
        finalOutFlowAtOutlet = [0]*totalIntervals       # List initialise for outflow values
        timeStamp = []      # List for time stamp corresponding to outflow values

        # Below is for finding outflow value at outlet node for a specific simulation time interval considering distance of the valley segment from outlet node
        for i in range(len(valleySegmentsCollectionOnIntervalTime)):
            for j in range(len(valleySegmentsCollectionOnIntervalTime[i])):
                finalOutFlowAtOutlet[i+valleyIndicatorNumber[j]-1] = finalOutFlowAtOutlet[i+valleyIndicatorNumber[j]-1] + valleySegmentsCollectionOnIntervalTime[i][j]

        # Below is for getting smooth outflow value
        for i in range(len(finalOutFlowAtOutlet)):
            timeStamp.append(i*deltaTime)
            if i>0 and i+1<len(finalOutFlowAtOutlet):
                finalOutFlowAtOutlet[i] = (finalOutFlowAtOutlet[i-1] + finalOutFlowAtOutlet[i] + finalOutFlowAtOutlet[i+1])/3

        ########################### CODE ENDS HERE: Another method for transfer of water from one valley node to another ###########################
        
        ########################### CODE STARTS HERE: Final hydrograph output along with max flow value and corresponding time ###########################

        maxFlowRate, timeAtMaxFlowRate = 0.0, 0     # Variables initialisation

        # It's totally choice which one to show for result, uncomment one code and comment another
        
        # Second method
        for i in range(len(finalOutFlowAtOutlet)):
            if finalOutFlowAtOutlet[i] > maxFlowRate:
                maxFlowRate = finalOutFlowAtOutlet[i]
                timeAtMaxFlowRate = timeStamp[i]
        
        data = {'Time Stamp (in sec)': timeStamp, 'Flow Rate (in m^3/s)': finalOutFlowAtOutlet}

        # # Kinematic Wave Approximation method
        # for i in range(len(finalXYGraphPoints[1])):
        #     if finalXYGraphPoints[1][i] > maxFlowRate:
        #         maxFlowRate = finalXYGraphPoints[1][i]
        #         timeAtMaxFlowRate = finalXYGraphPoints[0][i]

        # data = {'Time Stamp (in sec)': finalXYGraphPoints[0], 'Flow Rate (in m^3/s)': finalXYGraphPoints[1]}

        # Below is redundant code for showing graph using pyplot
        df = pd.DataFrame(data)

        figure = plt.figure(figsize=(5, 4), dpi=100)
        ax = figure.add_subplot(111)
        line = FigureCanvasTkAgg(figure, self)
        line.get_tk_widget().place(x=260, y=210, anchor=tk.CENTER)
        df = df[['Time Stamp (in sec)', 'Flow Rate (in m^3/s)']].groupby('Time Stamp (in sec)').sum()
        df.plot(kind='line', legend=True, ax=ax, color='b', marker='', linestyle='-', fontsize=10)
        ax.set_title('Outflow Hydrograph')
        # Show corresponding labels and its values
        ttk.Label(self, text= str(maxFlowRate)+' m^3/sec', foreground='red', font='Times 17 italic bold').place(x=520,y=150,anchor=tk.W)
        ttk.Label(self, text= str(timeAtMaxFlowRate)+' seconds', foreground='red', font='Times 17 italic bold').place(x=520,y=300,anchor=tk.W)

        self.deleteFilesFromLocalStorage()  # Delete all the files from local storage which were formed during operations
    
    def deleteFilesFromLocalStorage(self):
        # 12 '.txt' files for Nodes and Segments, 1 '.jpg' for contour map image formed during operation 
        os.remove(self.fileLocation + '/catchmentNodes.txt')
        os.remove(self.fileLocation + '/catchmentSegments.txt')
        os.remove(self.fileLocation + '/contourNodes.txt')
        os.remove(self.fileLocation + '/contourSegments.txt')
        os.remove(self.fileLocation + '/finalNodes.txt')
        os.remove(self.fileLocation + '/finalSegments.txt')
        os.remove(self.fileLocation + '/fullTransferSegments.txt')
        os.remove(self.fileLocation + '/noFlowSegments.txt')
        os.remove(self.fileLocation + '/ridgeNodes.txt')
        os.remove(self.fileLocation + '/ridgeSegments.txt')
        os.remove(self.fileLocation + '/valleyNodes.txt')
        os.remove(self.fileLocation + '/valleySegments.txt')
        os.remove(self.fileLocation + '/pointsCollection.jpg')



class VisualiseCatchmentAreaAndProceedForFinal(tk.Tk, ZoomInZoomOutFunctionality):

    def __init__(self, imageLoc, scaleFactor):
        
        # Initialise the wideget window propertiess and neccessary variables
        super().__init__()
        self.title('Visualise Catchment Area...')
        self.geometry('850x400')
        self.minsize(850,400)
        self.maxsize(850,400)
        # Get current working directory so that background image could be find out
        currentWorkingDirectory = os.getcwd()
        imageFileLocation = currentWorkingDirectory + '\i1.png'
        pathToIcon = currentWorkingDirectory + r'\guiIcon.ico'
        # Set icon for the window created
        self.wm_iconbitmap(pathToIcon)
        self.backgroundImageObject = tk.PhotoImage(file = imageFileLocation)
        # Get hold on previous contour-image used
        self.originalImageFileLocation = imageLoc
        self.imageLocation = imageLoc + '/pointsCollection.jpg'
        self.scaleFactor = scaleFactor
        # Create vertical separation and Set background image through canvas
        self.newCanvas = tk.Canvas(self)
        self.newCanvas.create_image(0,0,anchor=tk.NW,image=self.backgroundImageObject)
        self.newCanvas.create_line(425, 0, 425, 400)
        self.newCanvas.place(x=0,y=0,relheight=1,relwidth=1)

        # Set creative button properties
        ttk.Style().configure('VisualiseCatchmentAreaAndProceedForFinal.TButton', font=('Times italic bold', 17))
        ttk.Style().map('VisualiseCatchmentAreaAndProceedForFinal.TButton', foreground=[('pressed', 'blue'), ('active', 'red'), ('!disabled', 'red')], background=[('pressed', 'yellow'), ('active', 'black'), ('!disabled', 'yellow')])
        ttk.Style().configure('VisualiseCatchmentAreaAndProceedForFinal.TMenubutton', font=('Times italic bold', 15))
        ttk.Style().map('VisualiseCatchmentAreaAndProceedForFinal.TMenubutton', foreground=[('pressed', 'yellow'), ('active', 'red'), ('!disabled', 'black')], background=[('pressed', 'red'), ('active', 'yellow'), ('!disabled', 'white')])
        ttk.Style().configure('generalButton.TButton', font=('Times italic bold', 13))
        ttk.Style().map('generalButton.TButton', foreground=[('pressed', 'blue'), ('active', 'red'), ('!disabled', 'red')], background=[('pressed', 'yellow'), ('active', 'black'), ('!disabled', 'yellow')])
        
        # Button's design aspects
        ttk.Button(self, text='3D Visualise Catchment\n         (in Browser)      ', style='VisualiseCatchmentAreaAndProceedForFinal.TButton', command=self.openGraphInBrowser, padding=10).place(x=212,y=200,anchor=tk.CENTER)
        ttk.Button(self, text='Proceed to Hydrograph', style='VisualiseCatchmentAreaAndProceedForFinal.TButton', command=self.nextOperations, padding=10).place(x=638,y=240,anchor=tk.CENTER)
        ttk.Button(self, text='View Nodes', style='generalButton.TButton', command=self.viewNodes).place(x=638,y=110,anchor=tk.CENTER)

        # Initialise new menu for outlet point's input
        self.Node = tk.StringVar()
        self.menuButton = ttk.Menubutton(self, text='Select outlet node', style='VisualiseCatchmentAreaAndProceedForFinal.TMenubutton')
        self.menulist = tk.Menu(self.menuButton, tearoff=0)
        self.menuButton['menu'] = self.menulist
        self.menuButton.place(x=638,y=150,anchor=tk.CENTER)
        self.totalCatchmentNodes = 0
        self.canProceed = 0

        # # Design and positioning of Warning labels, below for position accuracy hence should be comment out during actual operation
        # self.nodeSelectionWarning = ttk.Label(self, text='test label !!!', foreground='red', font='Times 15 italic bold')
        # self.nodeSelectionWarning.place(x=638,y=190,anchor=tk.CENTER)
        # self.nodeAddWarning1 = ttk.Label(self, text='Please wait, processing...', foreground='red', font='Times 15 italic bold')
        # self.nodeAddWarning1.place(x=212,y=270,anchor=tk.CENTER)
        # self.nodeAddWarning2 = ttk.Label(self, text='Please wait, processing...', foreground='red', font='Times 15 italic bold')
        # self.nodeAddWarning2.place(x=638,y=290,anchor=tk.CENTER)

        # Outlet point should be one of the valley Nodes, Initilise the menulist for it
        try:

            newFileName = self.originalImageFileLocation + '/valleyNodes.txt'
            pointCloud = np.loadtxt(newFileName,skiprows=0,ndmin=2)
            nodes = pointCloud[:,0]

            for index in range(0, len(nodes)):
                newNumbering = 'N' + str(int(index+1))
                self.menulist.add_radiobutton(label=newNumbering, value=newNumbering, variable=self.Node, command=self.updateNode)

        except:
            # This indicates that there is no valley points so we can't proceed for hydrograph
            self.canProceed = 2

        # This variable will keep track whether our data is triangulated or not, 0 here means process has not been started
        self.isProcessing = 0


    def updateNode(self):
        # Update the option selected and show it
        self.menuButton['text'] = self.Node.get()
        # Since outlet point is selected, we can proceed for hydrograph
        self.canProceed = 1
        # Remove the previously existed label, if any
        labelToShow = 'None'
        self.selectNodeErrorHandling(labelToShow)


    def mouseHandlingForViewNodes(self, event, x, y, flags, param):
        # Only zoom-in/zoom-out function is our interest
        if(event==10):
            self.zoomInZoomOut(x, y, flags)
        else:
            return
        # Refresh the page after effects
        self.reconstructWindow()
        cv2.imshow('Entered Nodes...', self.newImage)


    def viewNodes(self):
        # Load image and put in the window
        self.newImage = cv2.imread(self.imageLocation, 1)
        self.setImageWindowDimensions()
        self.newImage = cv2.resize(self.newImage,self.resizedDimensions)
        cv2.imshow('Entered Nodes...', self.newImage)
        cv2.setMouseCallback('Entered Nodes...', self.mouseHandlingForViewNodes)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    
    def selectNodeErrorHandling(self, newLabel):
        # Remove any previous Label if exist
        try:
            self.nodeSelectionWarning.destroy()
        except:
            pass
        # Showing corresponding Label, if any
        if(newLabel != 'None'):
            self.nodeSelectionWarning = ttk.Label(self, text=newLabel, foreground='red', font='Times 15 italic bold')
            self.nodeSelectionWarning.place(x=638,y=190,anchor=tk.CENTER)


    def initialiseNodesAndSegments(self, fileName, nodeStore, segStore):
        # Get all requested nodes from fileName+Nodes.txt and put them in the nodeStore
        try:
            # Get content from the  file
            newFileName = self.originalImageFileLocation + fileName + 'Nodes.txt'
            pointCloud = np.loadtxt(newFileName,skiprows=0,ndmin=2)
            xyz = pointCloud[:,:]

            # Set content as per the our format
            for ele in xyz:
                # Order: first three are xyz coordinate and last number is currently set 1 which means this perticular node is active and would be 
                # under consideration, this can be made 0 (inactive) when same node is there in two different set (say present in both valley as well
                # as catchment boundary) in that case one will be acted as high priority and lowest prioritised set have to eliminate that node
                nodeStore.append([ele[0], ele[1], ele[2], 1])

        except:
            # File is either empty or doesn't exist
            pass

        # Get all requested segments from fileName+Segments.txt and put them in the segStore
        try:
            # Get content from the file
            newFileName = self.originalImageFileLocation + fileName + 'Segments.txt'
            pointCloud = np.loadtxt(newFileName,skiprows=0,ndmin=2)
            tempSegments = pointCloud[:,:]

            # Set content as per our format
            for element in tempSegments:
                # First four are from file and last is for active/inactive bit. Segment will be said as inactive when we current segment is intersecting 
                # with the other set's segment and we have to find point of intersection and have to add appropriate new segmentd and have to eliminate 
                # original segment by making this bit set as zero. 
                # Key Note: When new intersection point will be found out from intersection of two segments, this new node will be inserted into highest 
                # prioritised set, but we have to mark this segment in both set so here 3rd and 4th element will be useful to show exactly from where we 
                # should look for that node in that segment. 
                segStore.append([int(element[0])-1, int(element[1])-1, int(element[2]), int(element[3]), 1])

        except:
            # File is either empty or doesn't exist
            pass


    def createNewNodeList(self, oldList, newList, subList):
        # If some nodes are overlapping then have to eliminate them and appropriate action has to be carried out
        for i in range(len(oldList)):
            # If node is inactive, don't put in newList and add one to subList in that perticular node index
            if oldList[i][3] == 0:
                subList[i] = 1
            # Otherwise append to the newList because it is active node
            else:
                newList.append([oldList[i][0], oldList[i][1], oldList[i][2]])
            
            # Pre-sum has to be maintain so that by subtracking this number to original node number will gives correct order of the active node 
            # number index. Eliminated nodes already get adjusted in the segment's formation so don't need any special attention here.
            if i>0:
                subList[i] = subList[i] + subList[i-1]

    
    def refineSegmentList(self, segList, subList, indNum):
        # As per described in method named 'createNewNodeList', subtraction part is done in this method. indNum is indicating variable which shows 
        # which perticular type of nodes we are currently interested in.
        for i in range(len(segList)):
            if segList[i][2] == indNum:
                segList[i][0] = segList[i][0] - subList[segList[i][0]]
            if segList[i][3] == indNum:
                segList[i][1] = segList[i][1] - subList[segList[i][1]]


    def doubleDeactivateRedundantNodes(self, highNodeList, lowNodeList, firstSegment, secondSegment, highIndNum, lowIndNum):
        # Remove all the overlapping nodes in the highNodeList and lowNodeList from the lowNodeList
        for i in range(len(highNodeList)):
            for j in range(len(lowNodeList)):
                # Process should get continue after conforming that it is overlapping node case
                if (highNodeList[i][0] == lowNodeList[j][0]) and (highNodeList[i][1] == lowNodeList[j][1]):
                    lowNodeList[j][3] = 0   # Make lowNodeList's node inactive

                    # Iterate over all the segments present in the firstSegment and if any of them is inactivated node from the lowNodeList (this 
                    # is confirmed by comparing with the lowIndNum) then change its index number to highNodeList and set priority number belong to highNodeList
                    for k in range(len(firstSegment)):
                        if j == firstSegment[k][0] and firstSegment[k][2] == lowIndNum:
                            firstSegment[k][0] = i
                            firstSegment[k][2] = highIndNum

                        if j == firstSegment[k][1] and firstSegment[k][3] == lowIndNum:
                            firstSegment[k][1] = i
                            firstSegment[k][3] = highIndNum

                    # Similarly we should check lowNodeList inactive node from the secondSegment
                    for k in range(len(secondSegment)):
                        if j == secondSegment[k][0] and secondSegment[k][2] == lowIndNum:
                            secondSegment[k][0] = i
                            secondSegment[k][2] = highIndNum
                        
                        if j == secondSegment[k][1] and secondSegment[k][3] == lowIndNum:
                            secondSegment[k][1] = i
                            secondSegment[k][3] = highIndNum


    def singleDeactivateRedundantNodes(self, highNodeList, lowNodeList, segList, highIndNum, lowIndNum):
        # Only difference between this method and the 'doubleDeactivateRedundantNodes' is that, here we have only one segment list named 'segList'
        for i in range(len(highNodeList)):
            for j in range(len(lowNodeList)):
                if (highNodeList[i][0] == lowNodeList[j][0]) and (highNodeList[i][1] == lowNodeList[j][1]):
                    lowNodeList[j][3] = 0

                    for k in range(len(segList)):
                        if j == segList[k][0] and segList[k][2] == lowIndNum:
                            segList[k][0] = i
                            segList[k][2] = highIndNum
                        
                        if j == segList[k][1] and segList[k][3] == lowIndNum:
                            segList[k][1] = i
                            segList[k][3] = highIndNum


    def findPointOfIntersections(self, catchmentNodes, valleyNodes, ridgeNodes, contourNodes, highSegments, lowSegments, highIndNum):
        # This method is for the finding the intersection point between the segments of highSegments and lowSegments
        # Iterate over high priority segments and then check against low priority segments whether there is any possibility of point of intersection
        cnt1 = -1   # This variable is for the indexing of the highSegments
        for seg in highSegments:
            cnt1 = cnt1 + 1     # Increment after each iteration

            # If we encounter inactive segment we should escape remaining process
            if seg[4] == 0:
                continue
            
            # Equation of a line considered: y_mult1*y = x_mult1*x + c_const1
            x_mult1, y_mult1, c_const1 = 1.0, 1.0, 0.0
            pt1, pt2 = [], []       # Store in [x, y]
            pt1z, pt2z = 0.0, 0.0   # z coordinate of points

            # Since different type of nodes are intermixed in segments, we have to check from which node list, current node is taken out
            # As per previous convention: 0-> catchment type node, 1-> valley type node, 2-> ridge type node, 3-> contour type node
            # Below is for the first/start node of segment
            if seg[2] == 0:
                pt1 = [catchmentNodes[seg[0]][0], catchmentNodes[seg[0]][1]]
                pt1z = catchmentNodes[seg[0]][2]
            elif seg[2] == 1:
                pt1 = [valleyNodes[seg[0]][0], valleyNodes[seg[0]][1]]
                pt1z = valleyNodes[seg[0]][2]
            elif seg[2] == 2:
                pt1 = [ridgeNodes[seg[0]][0], ridgeNodes[seg[0]][1]]
                pt1z = ridgeNodes[seg[0]][2]
            else:
                pt1 = [contourNodes[seg[0]][0], contourNodes[seg[0]][1]]
                pt1z = contourNodes[seg[0]][2]

            # Similarly, Below is for the second/end node of segment
            if seg[3] == 0:
                pt2 = [catchmentNodes[seg[1]][0], catchmentNodes[seg[1]][1]]
                pt2z = catchmentNodes[seg[1]][2]
            elif seg[3] == 1:
                pt2 = [valleyNodes[seg[1]][0], valleyNodes[seg[1]][1]]
                pt2z = valleyNodes[seg[1]][2]
            elif seg[3] == 2:
                pt2 = [ridgeNodes[seg[1]][0], ridgeNodes[seg[1]][1]]
                pt2z = ridgeNodes[seg[1]][2]
            else:
                pt2 = [contourNodes[seg[1]][0], contourNodes[seg[1]][1]]
                pt2z = contourNodes[seg[1]][2]

            # Find the appropriate value of the x_mult1, y_mult1 and c_const1
            if pt1[0] == pt2[0]:    # When there is vertical segment
                x_mult1 = -1.0
                y_mult1 = 0.0
                c_const1 = pt1[0]
            elif pt1[1] == pt2[1]:  # When there is horizontal segment
                x_mult1 = 0.0
                c_const1 = pt1[1]
            else:                   # when it is inclined at some angle
                x_mult1 = (pt2[1]-pt1[1])/(pt2[0]-pt1[0])
                c_const1 = y_mult1*pt1[1] - x_mult1*pt1[0]

            # Iterate over low priority segments
            cnt2 = -1   # This variable is for the indexing of the lowSegments
            for j in range(len(lowSegments)):
                # Below are having same meaning as already mentioned above
                locSeg = lowSegments[j]
                cnt2 = cnt2 + 1

                if locSeg[4] == 0:
                    continue

                x_mult2, y_mult2, c_const2 = 1.0, 1.0, 0.0
                pt3, pt4 = [], []

                if locSeg[2] == 0:
                    pt3 = [catchmentNodes[locSeg[0]][0], catchmentNodes[locSeg[0]][1]]
                elif locSeg[2] == 1:
                    pt3 = [valleyNodes[locSeg[0]][0], valleyNodes[locSeg[0]][1]]
                elif locSeg[2] == 2:
                    pt3 = [ridgeNodes[locSeg[0]][0], ridgeNodes[locSeg[0]][1]]
                else:
                    pt3 = [contourNodes[locSeg[0]][0], contourNodes[locSeg[0]][1]]

                if locSeg[3] == 0:
                    pt4 = [catchmentNodes[locSeg[1]][0], catchmentNodes[locSeg[1]][1]]
                elif locSeg[3] == 1:
                    pt4 = [valleyNodes[locSeg[1]][0], valleyNodes[locSeg[1]][1]]
                elif locSeg[3] == 2:
                    pt4 = [ridgeNodes[locSeg[1]][0], ridgeNodes[locSeg[1]][1]]
                else:
                    pt4 = [contourNodes[locSeg[1]][0], contourNodes[locSeg[1]][1]]

                if pt3[0] == pt4[0]:
                    x_mult2 = -1.0
                    y_mult2 = 0.0
                    c_const2 = pt3[0]
                elif pt3[1] == pt4[1]:
                    x_mult2 = 0.0
                    c_const2 = pt3[1]
                else:
                    x_mult2 = (pt4[1]-pt3[1])/(pt4[0]-pt3[0])
                    c_const2 = y_mult2*pt3[1] - x_mult2*pt3[0]

                # Below two are indicator variables. In f1, high priority segment is forming a line and try to check what is position of low priority 
                # nodes wrt this line. Similarly in f2, low priority segmnt is forming a line and try to check what is position of high priority nodes 
                # wrt this line. By position we mean whether both nodes lies on same side of line (then there will be no point of intersection) or both 
                # lies on opposite side of line (Here is possibility of intersection) or one (or both) node lies on the line (Here is also possibility of 
                # intersection).
                f1 = (y_mult1*pt3[1]-x_mult1*pt3[0]-c_const1)*(y_mult1*pt4[1]-x_mult1*pt4[0]-c_const1)
                f2 = (y_mult2*pt1[1]-x_mult2*pt1[0]-c_const2)*(y_mult2*pt2[1]-x_mult2*pt2[0]-c_const2)

                if f1 > 0 or f2 > 0:    # Points lies on same side of the line hence no possibility of intersection
                    continue
                elif (pt1 == pt3) or (pt1 == pt4) or (pt2 == pt3) or (pt2 == pt4):  # End points are same, then no need of any further checkup
                    continue
                elif  (x_mult1 == x_mult2): # Same slope also don't allow intersection
                    continue
                else:       # Intersection is present hence do further process
                    # Find the point of intersection
                    mat1 = np.array([[y_mult1, -x_mult1], [y_mult2, -x_mult2]])
                    mat2 = np.array([c_const1, c_const2])
                    sol = np.linalg.solve(mat1, mat2)
                    x_sol, y_sol = sol[1], sol[0]

                    if x_sol == pt1[0] and y_sol == pt1[1]:     # If high priority segment's start point is intersection 
                        lowSegments[cnt2][4] = 0    # Inactivate low segment
                        # Since low priotity segment is partitioned, we have to add two new segments considering one end node is high priority segment's 
                        # start point and other from one of the two end points of the low priority segment for each. 
                        lowSegments.append([locSeg[0], seg[0], locSeg[2], seg[2], 1])
                        lowSegments.append([seg[0], locSeg[1], seg[2], locSeg[3], 1])
                    elif x_sol == pt2[0] and y_sol == pt2[1]:   # If high priority segment's end point is intersection
                        lowSegments[cnt2][4] = 0    # Inactivate low segment
                        # Since low priority segment is partitioned, we have to add two new segments considering one end node is high priority segment's 
                        # end point and other from one of the two end points of the low priority segment for each. 
                        lowSegments.append([locSeg[0], seg[1], locSeg[2], seg[3], 1])
                        lowSegments.append([seg[1], locSeg[1], seg[3], locSeg[3], 1])
                    elif x_sol == pt3[0] and y_sol == pt3[1]:   # If low priority segment's start point is intersection
                        highSegments[cnt1][4] = 0   # Inactivate high segment
                        # Since high priority segment is partitioned, we have to add two new segments considering one end node is low priority segment's 
                        # start point and other from one of the two end points of the high priority segment for each. 
                        highSegments.append([seg[0], locSeg[0], seg[2], locSeg[2], 1])
                        highSegments.append([locSeg[0], seg[1], locSeg[2], seg[3], 1])
                        # Since high segment itself got partitioned we have to go on next high priority segment by breaking current low segments loop. 
                        # Reason: since high priority got partitioned we no longer take account of original high priority segment, anyway two newly 
                        # formed high priority segment would be there to account for other points of intersections
                        break
                    elif x_sol == pt4[0] and y_sol == pt4[1]:   # If low priority segment's end point is intersection
                        highSegments[cnt1][4] = 0   # Inactivate high segment
                        # Since high priority segment is partitioned, we have to add two new segments considering one end node is low priority segment's 
                        # end point and other from one of the two end points of the high priority segment for each. 
                        highSegments.append([seg[0], locSeg[1], seg[2], locSeg[3], 1])
                        highSegments.append([locSeg[1], seg[1], locSeg[3], seg[3], 1])
                        # Same reason as above we have to break low priority segments loop
                        break
                    else:       # Intersection point is not any end point
                        # Inactivate both low and high priority segments
                        highSegments[cnt1][4] = 0
                        lowSegments[cnt2][4] = 0
                        cnt3 = -1

                        # Idea here is that newly formed node is added to the high priority node list. For that check which is high priority node 
                        # list and then append the newly formed node there And also z-coordinate is mean of two high priority node's z-coordinate.
                        if highIndNum == 0:
                            catchmentNodes.append([x_sol, y_sol, (pt1z+pt2z)/2])
                            cnt3 = cnt3 + len(catchmentNodes)
                        elif highIndNum == 1:
                            valleyNodes.append([x_sol, y_sol, (pt1z+pt2z)/2])
                            cnt3 = cnt3 + len(valleyNodes)
                        elif highIndNum == 2:
                            ridgeNodes.append([x_sol, y_sol, (pt1z+pt2z)/2])
                            cnt3 = cnt3 + len(ridgeNodes)
                        else:
                            contourNodes.append([x_sol, y_sol, (pt1z+pt2z)/2])
                            cnt3 = cnt3 + len(contourNodes)
                        
                        # Total 4 new segments has formed hence add them in lowSegments and highSegments.
                        lowSegments.append([locSeg[0], cnt3, locSeg[2], highIndNum, 1])
                        lowSegments.append([cnt3, locSeg[1], highIndNum, locSeg[3], 1])
                        highSegments.append([seg[0], cnt3, seg[2], highIndNum, 1])
                        highSegments.append([cnt3, seg[1], highIndNum, seg[3], 1])
                        # Here also high priority segment got partitioned hence as we have to break low priority segment's loop. 
                        break


    def absorbHangingSegments(self, oldList, newList):
        # This method is to erase out all inactivated segments and next transfer only useful information i.e. remove active/inactive bit
        for element in oldList:
            if element[4] == 0:
                continue
            newList.append([element[0], element[1], element[2], element[3]])


    def prepareFinalSegmentList(self, catchmentListLength, valleyListLength, ridgeListLength, oldSegmentList, newSegmentList):
        # This mrthod is useful for making final refined segments of our interest, absolute in nature i.e. no further distinction between valley 
        # or ridge or catchment or contour nodes.
        # Make note:- Nodes will be finally added in the order: Catchment Nodes + Valley Nodes + Ridge Nodes + Contour Nodes

        for element in oldSegmentList:  # Iterate over old segments to refine them
            nextList = []   # Store end nodes absolute in nature

            if element[0] == element[1] and element[2] == element[3]:   # Segments which are making self loops should be rejected here
                continue
            
            # Start node is assigned absolute node index value
            if element[2] == 0:
                nextList.append(element[0])
            elif element[2] == 1:
                nextList.append(catchmentListLength + element[0])
            elif element[2] == 2:
                nextList.append(catchmentListLength + valleyListLength + element[0])
            else:
                nextList.append(catchmentListLength + valleyListLength + ridgeListLength + element[0])

            # End node is assigned absolute node index value
            if element[3] == 0:
                nextList.append(element[1])
            elif element[3] == 1:
                nextList.append(catchmentListLength + element[1])
            elif element[3] == 2:
                nextList.append(catchmentListLength + valleyListLength + element[1])
            else:
                nextList.append(catchmentListLength + valleyListLength + ridgeListLength + element[1])

            # append segment representing absolute nodes
            newSegmentList.append(nextList)


    def processInputNodesAndSegments(self):
        # Now state is changed from 'Not touched for triangulation' to 'Under process'
        self.isProcessing = 1
        
        # Initialise the list required for opertions
        tempCatchmentNodes = []
        catchmentSegments = []
        tempValleyNodes = []
        valleySegments = []
        tempRidgeNodes = []
        ridgeSegments = []
        tempContourNodes = []
        contourSegments = []

        # Get diffrent type of nodes and segments from local hardware
        self.initialiseNodesAndSegments('/catchment', tempCatchmentNodes, catchmentSegments)
        self.initialiseNodesAndSegments('/valley', tempValleyNodes, valleySegments)
        self.initialiseNodesAndSegments('/ridge', tempRidgeNodes, ridgeSegments)
        self.initialiseNodesAndSegments('/contour', tempContourNodes, contourSegments)

        # Remove all overlapping nodes, if exist
        self.doubleDeactivateRedundantNodes(tempCatchmentNodes, tempValleyNodes, valleySegments, ridgeSegments, 0, 1)
        self.singleDeactivateRedundantNodes(tempCatchmentNodes, tempRidgeNodes, ridgeSegments, 0, 2)
        self.singleDeactivateRedundantNodes(tempCatchmentNodes, tempContourNodes, contourSegments, 0, 3)

        self.doubleDeactivateRedundantNodes(tempContourNodes, tempValleyNodes, valleySegments, ridgeSegments, 3, 1)
        self.singleDeactivateRedundantNodes(tempContourNodes, tempRidgeNodes, ridgeSegments, 3, 2)

        # Store all non-overlapping nodes in below
        catchmentNodes = []
        valleyNodes = []
        ridgeNodes = []
        contourNodes = []

        # Since catchment nodes had highest priority, all its nodes will be preserved
        for node in tempCatchmentNodes:
            catchmentNodes.append([node[0], node[1], node[2]])

        # Erase out redundant nodes for valley lines, and refine valley & ridge segments on basis of that
        downIntervalList = [0]*len(tempValleyNodes)
        self.createNewNodeList(tempValleyNodes, valleyNodes, downIntervalList)
        self.refineSegmentList(valleySegments, downIntervalList, 1)
        self.refineSegmentList(ridgeSegments, downIntervalList, 1)

        # Similarly erase out redundant nodes from ridge line, and refine its segments
        downIntervalList = [0]*len(tempRidgeNodes)
        self.createNewNodeList(tempRidgeNodes, ridgeNodes, downIntervalList)
        self.refineSegmentList(ridgeSegments, downIntervalList, 2)

        # Here redundant contour nodes are removed, and then valley, ridge and contour segments are refined
        downIntervalList = [0]*len(tempContourNodes)
        self.createNewNodeList(tempContourNodes, contourNodes, downIntervalList)
        self.refineSegmentList(contourSegments, downIntervalList, 3)
        self.refineSegmentList(valleySegments, downIntervalList, 3)
        self.refineSegmentList(ridgeSegments, downIntervalList, 3)

        # Find point of intersections when high priority segments are catchment boundary
        self.findPointOfIntersections(catchmentNodes, valleyNodes, ridgeNodes, contourNodes, catchmentSegments, valleySegments, 0)
        self.findPointOfIntersections(catchmentNodes, valleyNodes, ridgeNodes, contourNodes, catchmentSegments, ridgeSegments, 0)
        self.findPointOfIntersections(catchmentNodes, valleyNodes, ridgeNodes, contourNodes, catchmentSegments, contourSegments, 0)

        # Find point of intersections when high priority segments are contour line's segment
        self.findPointOfIntersections(catchmentNodes, valleyNodes, ridgeNodes, contourNodes, contourSegments, valleySegments, 3)
        self.findPointOfIntersections(catchmentNodes, valleyNodes, ridgeNodes, contourNodes, contourSegments, ridgeSegments, 3)

        # Remove all the deactivated segments
        tempSegments = []
        self.absorbHangingSegments(catchmentSegments, tempSegments)
        catchmentSegments = tempSegments

        tempSegments = []
        self.absorbHangingSegments(valleySegments, tempSegments)
        valleySegments = tempSegments

        tempSegments = []
        self.absorbHangingSegments(ridgeSegments, tempSegments)
        ridgeSegments = tempSegments

        tempSegments = []
        self.absorbHangingSegments(contourSegments, tempSegments)
        contourSegments = tempSegments

        # Final useful storage for the nodes and segments
        finalNodes = []     # All nodes would be stored here in sequence to make it absolute accessible list
        finalSegments = []  # All segments will be stored here which are mapped in nodes from 'finalNodes'
        finalNoFlowSegments = []    # These are actually ridge and catchment boundary segments
        finalFullTransferSegments = []  # These are valley segments
        
        # Order in which nodes will be stored: catchment nodes + valley nodes + ridge nodes + contour nodes
        finalNodes.extend(catchmentNodes), finalNodes.extend(valleyNodes), finalNodes.extend(ridgeNodes), finalNodes.extend(contourNodes)

        # Stores segments as per our requirement
        self.prepareFinalSegmentList(len(catchmentNodes), len(valleyNodes), len(ridgeNodes), catchmentSegments, finalNoFlowSegments)
        self.prepareFinalSegmentList(len(catchmentNodes), len(valleyNodes), len(ridgeNodes), valleySegments, finalFullTransferSegments)
        self.prepareFinalSegmentList(len(catchmentNodes), len(valleyNodes), len(ridgeNodes), ridgeSegments, finalNoFlowSegments)
        self.prepareFinalSegmentList(len(catchmentNodes), len(valleyNodes), len(ridgeNodes), contourSegments, finalSegments)

        # Final segments are full transfer segments and no transfer segments
        finalSegments.extend(finalNoFlowSegments), finalSegments.extend(finalFullTransferSegments)
        self.totalCatchmentNodes = len(catchmentNodes)

        # Store 'finalNodes' on local storage
        fileLocation = self.originalImageFileLocation + '/finalNodes.txt'
        with open(fileLocation, 'w') as file:
            for elements in finalNodes:
                file.write(str((float(elements[0]))*self.scaleFactor) + ' ' + str((float(elements[1]))*self.scaleFactor) + ' ' + str(float(elements[2])) + '\n')
            file.close()

        # Similarly store 'noFlowSegments' on local storage
        fileLocation = self.originalImageFileLocation + '/noFlowSegments.txt'
        with open(fileLocation, 'w') as file:
            for elements in finalNoFlowSegments:
                tempList = [elements[0], elements[1]]
                tempList.sort()
                file.write(str(tempList[0]) + ' ' + str(tempList[1]) + '\n')
            file.close()

        # And store 'fullTransferSegments' on local storage
        fileLocation = self.originalImageFileLocation + '/fullTransferSegments.txt'
        with open(fileLocation, 'w') as file:
            for elements in finalFullTransferSegments:
                tempList = [elements[0], elements[1]]
                tempList.sort()
                file.write(str(tempList[0]) + ' ' + str(tempList[1]) + '\n')
            file.close()

        # And store 'finalSegments' on local storage
        fileLocation = self.originalImageFileLocation + '/finalSegments.txt'
        with open(fileLocation, 'w') as file:
            for elements in finalSegments:
                tempList = [elements[0], elements[1]]
                tempList.sort()
                file.write(str(tempList[0]) + ' ' + str(tempList[1]) + '\n')
            file.close()

        # Now state is changed from 'Under process' to 'Process has completed'
        self.isProcessing = 2


    def openGraphInBrowser(self):
        # Before showing geographical inputs in the form of suface formed, we have to process inputs with no redundants nodes and segments should 
        # not cross each other without intersection, hence initial level data refinement has to be done before triangulation process
        
        # If refinement has not done yet, go ahead for that before proceed next
        if self.isProcessing == 0:
            self.nodeAddWarning1 = ttk.Label(self, text='Please wait, processing...', foreground='red', font='Times 15 italic bold')
            self.nodeAddWarning1.place(x=212,y=270,anchor=tk.CENTER)
            self.processInputNodesAndSegments()
            self.nodeAddWarning1.destroy()

        # If refinement process is under process, we should wait till it get completed
        elif self.isProcessing == 1:
            self.nodeAddWarning1 = ttk.Label(self, text='Please wait, still processing...', foreground='red', font='Times 15 italic bold')
            self.nodeAddWarning1.place(x=212,y=270,anchor=tk.CENTER)
            while(self.isProcessing == 1):
                pass
            self.nodeAddWarning1.destroy()
        
        # Get absolute nodes from the local storage
        newFileName = self.originalImageFileLocation + '/finalNodes.txt'
        pointCloud = np.loadtxt(newFileName,skiprows=0,ndmin=2)
        xyz = pointCloud[:,:]
        x, y, z = pointCloud[:,0], pointCloud[:,1], pointCloud[:,2]
        xyFormat = []   # We are triangulating the data in 2D format
        nodeSegments = []
        
        for ele in xyz:
            xyFormat.append([ele[0], ele[1]])

        # Similarly get absolute segments from local storage
        newFileName = self.originalImageFileLocation + '/finalSegments.txt'
        pointCloud = np.loadtxt(newFileName,skiprows=0,ndmin=2)
        segs = pointCloud[:,:]

        for ele in segs:
            nodeSegments.append((int(ele[0]), int(ele[1])))

        # Initiate triangulation process and store data in 'triangulatedData' variable
        triangulatedData = tr.triangulate({'vertices':xyFormat, 'segments':nodeSegments}, 'pA')

        # Below are three nodes of a triangles fromed from triangulation
        i_, j_, k_ = triangulatedData['triangles'][:,0], triangulatedData['triangles'][:,1], triangulatedData['triangles'][:,2]

        # Showing geometry using graph object module
        fig = go.Figure(data=[go.Mesh3d(x=x, y=y, z=z, i=i_, j=j_, k=k_, color='cyan', opacity=0.80)])
        f2 = go.FigureWidget(fig)
        f2.show()


    def nextOperations(self):
        # Before proceeding to the hydrograph final output page, do initial level refinement process as it was required in above method called 'openGraphInBrowser'
        
        # If there is no valley lines inputs given, we don't have any outlet node to select
        if (self.canProceed == 2):
            labelToShow = 'There is No valley Points.'
            self.selectNodeErrorHandling(labelToShow)
            return
        
        # Select outlet node corresponding to which outflow will be shown
        elif (self.canProceed == 0):
            labelToShow = 'Please select outlet node!'
            self.selectNodeErrorHandling(labelToShow)
            return
        
        # If refinement has not done yet, go ahead for that before proceed next 
        elif self.isProcessing == 0:
            self.nodeAddWarning2 = ttk.Label(self, text='Please wait, processing...', foreground='red', font='Times 15 italic bold')
            self.nodeAddWarning2.place(x=638,y=290,anchor=tk.CENTER)
            self.processInputNodesAndSegments()
            self.nodeAddWarning2.destroy()

        # If refinement process is under process, we should wait till it get completed
        elif self.isProcessing == 1:
            self.nodeAddWarning2 = ttk.Label(self, text='Please wait, still processing...', foreground='red', font='Times 15 italic bold')
            self.nodeAddWarning2.place(x=638,y=290,anchor=tk.CENTER)
            while(self.isProcessing == 1):
                pass
            self.nodeAddWarning2.destroy()

        # Get input from outlet node selected
        strToNum = self.Node.get()
        finalOutletNode = self.totalCatchmentNodes + int(strToNum[1:]) - 1
        fileLocations = self.originalImageFileLocation
        self.destroy()

        # Proceed to next page
        nextPage = ProcessSurfaceFlowAndFinalGraphOutput(fileLocations, finalOutletNode)
        nextPage.mainloop()



class ElevationValueInputFromWindow(tk.Toplevel):
    # This is sub-window for getting elevation value from user during catchment boundary's points collection. 

    def __init__(self, parent):
        # Set top level window on the previous defined window
        super().__init__(parent)

        # Set icon for this top-level window
        self.currentWorkingDirectory = os.getcwd()
        self.pathToIcon = self.currentWorkingDirectory + r'\guiIcon.ico'
        self.wm_iconbitmap(self.pathToIcon)
        self.minsize(250,120)
        self.maxsize(250,120)
        self.title("Elevation value")

        # If window is tried to closed by cross icon, follwing method will be invoke
        self.protocol("WM_DELETE_WINDOW", self.testAndProceed)
        self.elevationFromWindow = parent.elevationFromWindow

        # Page's design labels
        ttk.Label(self, text='Elevation: ', foreground='black', font='Times 15 italic bold').place(x=120, y=30, anchor=tk.E)
        ttk.Entry(self, textvariable=self.elevationFromWindow, foreground='black', font='helvetica 13', width=10).place(x=130,y=30,anchor=tk.W)
        # self.warningLabel = ttk.Label(self, text='test label !!!', foreground='red', font='Times 13 italic bold')
        # self.warningLabel.place(x=125,y=60,anchor=tk.CENTER)
        ttk.Button(self, text='Enter', command=self.testAndProceed).place(x=125,y=90,anchor=tk.CENTER)
        

    def testAndProceed(self):
        # Previous exsiting label should be destroyed before putting new or not required to put
        try:
            self.warningLabel.destroy()
        except:
            pass
        
        # Get value from the tkinter variable
        strVar = self.elevationFromWindow.get()

        # If we have not provided any value for elevation
        if(strVar == ''):
            self.warningLabel = ttk.Label(self, text="Enter a numerical value.", foreground='red', font='Times 13 italic bold')
            self.warningLabel.place(x=125,y=60,anchor=tk.CENTER)
            return

        # Handle corresponding error due to different reasons
        try:
            isFloatable = float(self.elevationFromWindow.get())     # Convert inputed value into float if possible
            
            if (isFloatable <= 0):
                self.warningLabel = ttk.Label(self, text="Enter a value > 0.", foreground='red', font='Times 13 italic bold')
                self.warningLabel.place(x=125,y=60,anchor=tk.CENTER)
                return
        except:
            self.warningLabel = ttk.Label(self, text="Elevation can't be Non-numerical!", foreground='red', font='Times 13 italic bold')
            self.warningLabel.place(x=125,y=60,anchor=tk.CENTER)
            return
        
        # If user input for elevation is as per our requirement we can exit from sub-window
        self.quit()
        self.destroy()



class CommonSubWindowProperties(tk.Toplevel):
    # Some properties, methods or variables which are common to some sub-windows are listed in this class so that they can be used by inheriting this class

    def __init__(self, parent):
        # Set top level window on the previous defined window
        super().__init__(parent)

        # Set icon for this top-level window
        self.currentWorkingDirectory = os.getcwd()
        self.pathToIcon = self.currentWorkingDirectory + r'\guiIcon.ico'
        self.wm_iconbitmap(self.pathToIcon)

        # If window is closed by cross icon
        self.protocol("WM_DELETE_WINDOW", self.dismissAndExit)

        # All properties of button for this page
        ttk.Style().configure('TButton', padding=2, relief="flat")
        ttk.Style().configure('generalButton.TButton', font=('Times italic bold', 13))
        ttk.Style().map('generalButton.TButton', foreground=[('pressed', 'blue'), ('active', 'red'), ('!disabled', 'red')], background=[('pressed', 'yellow'), ('active', 'black'), ('!disabled', 'yellow')])
        ttk.Style().configure('intermediateButton.TButton', font=('Times italic bold', 15))
        ttk.Style().map('intermediateButton.TButton', foreground=[('pressed', 'blue'), ('active', 'black'), ('!disabled', 'black')], background=[('pressed', 'red'), ('active', 'black'), ('!disabled', 'black')])
        ttk.Style().configure('navButton.TButton', font=('Times italic bold', 18))
        ttk.Style().map('navButton.TButton', foreground=[('pressed', 'blue'), ('active', 'black'), ('!disabled', 'black')], background=[('pressed', 'red'), ('active', 'black'), ('!disabled', 'black')])

        # Listing image
        self.imageLocation = parent.originalImageFileLocation + '/pointsCollection.jpg'
        self.originalImageFileLocation = parent.originalImageFileLocation
        self.destinationFolder = self.originalImageFileLocation + '/nocombosvposrp'

        # Create a copy of contour image so that in case later we want to reject the new changes made, we can delete modified image and make 
        # available this image at that position to make appearance as nothing got saved
        self.absoluteFileLocation = os.getcwd()
        os.chdir(self.originalImageFileLocation)
        os.mkdir('nocombosvposrp')
        os.chdir('nocombosvposrp')
        # self.destinationFolder = os.getcwd()
        sop.copy(self.imageLocation, self.destinationFolder)
        os.chdir(self.absoluteFileLocation)

    def dismissAndExit(self):
        self.quit()
        self.destroy()



class ValleyLinesWindow(CommonSubWindowProperties, ZoomInZoomOutFunctionality):
    # From inputs of nodes to segments formation would be done here in this class for the valley lines

    def __init__(self, parent):
        # Set top level window on the previous defined window
        super().__init__(parent)

        # Fix new geometry for this top-level window
        self.geometry('500x540')
        self.title('Valley Lines')

        # Line Separators in the design
        self.newCanvas = tk.Canvas(self)
        self.newCanvas.create_line(20, 200, 480, 200, dash=(3,1))
        self.newCanvas.create_line(0, 440, 500, 440, dash=(5,1))
        self.newCanvas.create_line(250, 440, 250, 540, dash=(1,1))
        self.newCanvas.place(x=0,y=0,relheight=1,relwidth=1)

        # Step 1 Design
        ttk.Label(self, text="Step 1: Select all nodes of valley line's graph", foreground='black', font='Times 17 italic bold').place(x=250, y=50, anchor=tk.CENTER)
        ttk.Button(self, text='Select Node', style='generalButton.TButton', command=self.selectNewNode).place(x=125,y=90,anchor=tk.CENTER)
        ttk.Label(self, text="Elevation:", foreground='black', font='Times 15 italic bold').place(x=250, y=90, anchor=tk.W)
        self.elevationValue = tk.StringVar()
        ttk.Entry(self, textvariable=self.elevationValue, foreground='black', font='helvetica 13', width=10).place(x=400,y=90,anchor=tk.CENTER)
        # self.nodeAddWarning = ttk.Label(self, text='test label !!!', foreground='black', font='Times 13 italic bold')
        # self.nodeAddWarning.pack(x=400,y=120,anchor=tk.CENTER)
        ttk.Button(self, text='Add Node', style='intermediateButton.TButton', command=self.addNewNode).place(x=250,y=150,anchor=tk.CENTER)

        # Step 2 Design
        ttk.Label(self, text="Step 2: connect nodes of valley line's graph", foreground='black', font='Times 17 italic bold').place(x=250, y=250, anchor=tk.CENTER)
        ttk.Button(self, text='View Nodes', style='generalButton.TButton', command=self.viewNodes).place(x=250,y=290,anchor=tk.CENTER)
        ttk.Label(self, text="Node 1: ", foreground='black', font='Times 15 italic bold').place(x=125, y=330, anchor=tk.E)
        ttk.Label(self, text="Node 2: ", foreground='black', font='Times 15 italic bold').place(x=375, y=330, anchor=tk.E)
        self.Node1 = tk.StringVar()
        self.Node2 = tk.StringVar()
        self.menuButton1 = ttk.Menubutton(self, text='Select first node')
        self.menulist1 = tk.Menu(self.menuButton1, tearoff=0)
        self.menuButton1['menu'] = self.menulist1
        self.menuButton1.place(x=125,y=330,anchor=tk.W)
        # self.nodeSelectionWarning1 = ttk.Label(self, text='test label !!!', foreground='black', font='Times 13 italic bold')
        # self.nodeSelectionWarning1.place(x=125,y=360,anchor=tk.W)
        self.menuButton2 = ttk.Menubutton(self, text='Select second node')
        self.menulist2 = tk.Menu(self.menuButton2, tearoff=0)
        self.menuButton2['menu'] = self.menulist2
        self.menuButton2.place(x=375,y=330,anchor=tk.W)
        # self.nodeSelectionWarning2 = ttk.Label(self, text='test label !!!', foreground='black', font='Times 13 italic bold')
        # self.nodeSelectionWarning2.place(x=375,y=360,anchor=tk.W)
        # self.nodeSelectionWarning3 = ttk.Label(self, text='test label !!!', foreground='black', font='Times 13 italic bold')
        # self.nodeSelectionWarning3.place(x=250,y=360,anchor=tk.CENTER)
        ttk.Button(self, text='Add Edge', style='intermediateButton.TButton', command=self.addNewEdge).place(x=250,y=390,anchor=tk.CENTER)
        
        # Bottom Section
        ttk.Button(self, text='Dismiss!', style='navButton.TButton', command=self.dismissAndExit).place(x=125,y=490,anchor=tk.CENTER)
        ttk.Button(self, text='Save!', style='navButton.TButton', command=self.saveAndExit).place(x=375,y=490,anchor=tk.CENTER)

        # Some important dictionaries initialisation
        # Below is used for the strong the node coordinates corresponding to a node label generated
        self.nodeDict = {}
        # Below is used for mapping of the joined nodes to a label generated for the segment formed
        self.segmentDict1 = {}
        # Below is reverse mapping of the above, where segment's label is mapped to joined nodes index
        self.segmentDict2 = {}

        # Below variable is just for storing of coordinates of the selected point on the contour map
        self.nodePoint = []

        # If we already have selected some valley nodes and segments previously, we have to initialise them in above dictionaries which will take 
        # account when we want to join current code to previously selected node and also error handling and a lot more
        try:
            # Get data from local storage for nodes
            newFileName = self.originalImageFileLocation + '/valleyNodes.txt'
            pointCloud = np.loadtxt(newFileName,skiprows=0,ndmin=2)
            nodes, nodeCoordinates = pointCloud[:,0], pointCloud[:,0:]
            self.currentNumber = len(nodes)+1       # This number will later useful for naming when we select new valley nodes from the contour

            # Initialise the dictionaries
            for index in range(0, len(nodes)):
                newNumbering = 'N' + str(int(index+1))  # Genrate name from the Index of the node
                self.nodeDict[newNumbering] = nodeCoordinates[index]    # Assign coordinates
                # Option has given to select from menu list so have to assign these nodes to those menulist
                self.menulist1.add_radiobutton(label=newNumbering, value=newNumbering, variable=self.Node1, command=self.updateNode1)
                self.menulist2.add_radiobutton(label=newNumbering, value=newNumbering, variable=self.Node2, command=self.updateNode2)

            # Similarly here segments are stored from local storage
            newFileName = self.originalImageFileLocation + '/valleySegments.txt'
            pointCloud = np.loadtxt(newFileName,skiprows=0,ndmin=2)
            segments, nodesCombination = pointCloud[:,0], pointCloud[:,0:]
            self.currentNumber1 = len(segments)+1       # This will help for naming when new segments will be selected

            # Initialise the dixtionaries
            for index in range(0, len(segments)):
                n1, n2 = 'N' + str(int(nodesCombination[index][0])), 'N' + str(int(nodesCombination[index][1]))
                # Let we have N1 and N2 and corresponding S1 then stored mappings are: N1N2 -> S1 and S1 -> [1, 2]
                self.segmentDict2['S' + str(int(index+1))] = [n1, n2]
                self.segmentDict1[n1+n2] = 'S' + str(int(index+1))
        
        except:
            # We have to state that there is no already existing nodes and segments, and both node and segment will get label with numbering from 1
            self.currentNumber = 1
            self.currentNumber1 = 1


    def mouseHandlingForNodePoints(self, event, x, y, flags, param):
        # Whenever mouse movement is done following will be available for the operation

        if(event==10): # This is for zoom-in or zoom-out
            self.zoomInZoomOut(x, y, flags)
        
        elif event == cv2.EVENT_LBUTTONDOWN or event==cv2.EVENT_RBUTTONDOWN:    # This is the point selection on the contour map
            self.redefineScale()
            # Store node coordinates
            self.nodePoint.append(self.imageInitialX+(int(x*self.scaleFactorX)))
            self.nodePoint.append(self.imageInitialY+(int(y*self.scaleFactorY)))
            # Window should be closed after selecting the node
            cv2.destroyWindow('Please select node point...')
            return
        
        else:   # We only interested in above two operations, if anything other than scroll and left or right click is done, it should have no effect
            return
        
        # After zoom-in or out window has to be reconstruct to show live effect
        self.reconstructWindow()
        cv2.imshow('Please select node point...', self.newImage)


    def selectNewNode(self):
        # Erase out previous data
        self.nodePoint.clear()

        # Show image and take new node
        self.newImage = cv2.imread(self.imageLocation, 1)
        self.setImageWindowDimensions()
        self.newImage = cv2.resize(self.newImage,self.resizedDimensions)
        cv2.imshow('Please select node point...', self.newImage)
        cv2.setMouseCallback('Please select node point...', self.mouseHandlingForNodePoints)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def addNodeErrorHandling(self, newLabel):
        # Remove any previous Label if exist
        try:
            self.nodeAddWarning.destroy()
        except:
            pass
        # Showing corresponding Label if we want to show it
        if(newLabel != 'None'):
            self.nodeAddWarning = ttk.Label(self, text=newLabel, foreground='red', font='helvetica 13')
            self.nodeAddWarning.place(x=400,y=120,anchor=tk.CENTER)


    def updateNode1(self):  # Show selected value
        self.menuButton1['text'] = self.Node1.get()

    
    def updateNode2(self):  # Show selected value
        self.menuButton2['text'] = self.Node2.get()


    def addNewNode(self):
        # After click on the add node button, we should handle some errors if there any otherwise it should be saved

        try:
            # If value is conversionable to float then we are good to go
            self.elevation = float(self.elevationValue.get())
            
            if self.elevation <= 0:         # Value can't be negative or zero in any case
                labelToShow = 'Enter a value (>0)'
                self.addNodeErrorHandling(labelToShow)
            
            elif len(self.nodePoint) == 0:  # If we have not selected node point on the contour map
                labelToShow = 'Select a node point'
                self.addNodeErrorHandling(labelToShow)

            else:                           # Remove flag because we are good to go
                labelToShow = 'None'
                # Remove previously showed label 
                self.addNodeErrorHandling(labelToShow)
                # Draw a point sized circle to indicate that here node point was selected
                self.newImage = cv2.imread(self.imageLocation, 1)
                self.circleVariable = cv2.circle(self.newImage, (self.nodePoint[0], self.nodePoint[1]), 2, (0, 0, 255), -1)
                cv2.imwrite(self.imageLocation, self.circleVariable)
                # Cooresponding label to that point also has to be shown
                newNumbering = 'N' + str(self.currentNumber)
                self.currentNumber = self.currentNumber + 1
                self.textVariable = cv2.putText(self.newImage, newNumbering, (self.nodePoint[0], self.nodePoint[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
                cv2.imwrite(self.imageLocation, self.textVariable)
                # Append node point to the list
                self.nodePoint.append(self.elevation)
                # This node point has to be add to the menu list so that it could be selected during segment selection
                self.menulist1.add_radiobutton(label=newNumbering, value=newNumbering, variable=self.Node1, command=self.updateNode1)
                self.menulist2.add_radiobutton(label=newNumbering, value=newNumbering, variable=self.Node2, command=self.updateNode2)
                self.nodeDict[newNumbering] = copy.deepcopy(self.nodePoint) # It must be deep copy otherwise error would be there having same value everywhere
                self.nodePoint.clear()  # Clear present value
                self.elevationValue.set('') # Reset elevation value to empty

        except:
            # If value entered is not float show corresponding error and put flag to reset it
            labelToShow = 'Enter a float value.'
            self.addNodeErrorHandling(labelToShow)


    def mouseHandlingForViewNodes(self, event, x, y, flags, param):
        # Only required functionality is zoom-in or out for better clear vision

        if(event==10): # This is for zoom-in or zoom-out
            self.zoomInZoomOut(x, y, flags)

        else:
            return
        
        # Reconstruct to have live effects
        self.reconstructWindow()
        cv2.imshow('Entered Nodes...', self.newImage)


    def viewNodes(self):
        # Load the contour map image where all the nodes are impressed and provided mouse handling
        self.newImage = cv2.imread(self.imageLocation, 1)
        self.setImageWindowDimensions()
        self.newImage = cv2.resize(self.newImage,self.resizedDimensions)
        cv2.imshow('Entered Nodes...', self.newImage)
        cv2.setMouseCallback('Entered Nodes...', self.mouseHandlingForViewNodes)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def selectNode1ErrorHandling(self, newLabel):
        # Remove label if any exist previously
        try:
            self.nodeSelectionWarning1.destroy()
        except:
            pass
        
        # Put appropriate label if any required
        if newLabel != 'None':
            self.nodeSelectionWarning1 = ttk.Label(self, text=newLabel, foreground='red', font='helvetica 13')
            self.nodeSelectionWarning1.place(x=125,y=360,anchor=tk.W)


    def selectNode2ErrorHandling(self, newLabel):
        # Remove label if any exist previously
        try:
            self.nodeSelectionWarning2.destroy()
        except:
            pass
        
        # Put appropriate label if any required
        if newLabel != 'None':
            self.nodeSelectionWarning2 = ttk.Label(self, text=newLabel, foreground='red', font='helvetica 13')
            self.nodeSelectionWarning2.place(x=375,y=360,anchor=tk.W)


    def selectNode3ErrorHandling(self, newLabel):
        # Remove label if any exist previously
        try:
            self.nodeSelectionWarning3.destroy()
        except:
            pass
        
        # Put appropriate label if any required
        if newLabel != 'None':
            self.nodeSelectionWarning3 = ttk.Label(self, text=newLabel, foreground='red', font='helvetica 13')
            self.nodeSelectionWarning3.place(x=250,y=360,anchor=tk.CENTER)


    def addNewEdge(self):
        # After selecting nodes from the menu list, some error handling have to done before making that segment or edge
        # Get what nodes are selected
        str1 = self.Node1.get()
        str2 = self.Node2.get()

        # Erase out any previously existing error label, if any present there
        labelToShow = 'None'
        self.selectNode1ErrorHandling(labelToShow)
        self.selectNode2ErrorHandling(labelToShow)
        self.selectNode3ErrorHandling(labelToShow)

        # Basic error handling when only one node is selected as end nodes of segment or not even one Or selected nodes are same
        if str1 == '':
            labelToShow = 'Select a node'
            self.selectNode1ErrorHandling(labelToShow)
            return
        
        elif str2 == '':
            labelToShow = 'Select a node'
            self.selectNode2ErrorHandling(labelToShow)
            return
        
        elif str1 == str2:
            labelToShow = 'Select two different nodes'
            self.selectNode3ErrorHandling(labelToShow)
            return

        segmentEnds = []    # Store nodes in sorted order in this lsit
        # Below is for sorted order
        if str1 < str2:
            segmentEnds.append(str1)
            segmentEnds.append(str2)
        else:
            segmentEnds.append(str2)
            segmentEnds.append(str1)

        # In case selected nodes already have a segment or edge defined previously
        try:
            ifexistPrevious = self.segmentDict1[segmentEnds[0]+segmentEnds[1]]
            labelToShow = f'This segment already exist: {ifexistPrevious}'
            self.selectNode3ErrorHandling(labelToShow)
            return  # Error has occured hence has to be return from this position
        except:
            pass    # Segment is not previously defined hence go ahead
        
        # Make label for this segment
        numbering = 'S' + str(self.currentNumber1)
        self.currentNumber1 = self.currentNumber1 + 1   # Increment for next operation

        # Save this segment in both dictionaries for accessibility
        self.segmentDict1[segmentEnds[0] + segmentEnds[1]] =  numbering
        self.segmentDict2[numbering] = segmentEnds

        # Mark this segment on the contour map along with corresponding label for this segment
        self.newImage = cv2.imread(self.imageLocation, 1)
        self.lineVariable = cv2.line(self.newImage, (int(self.nodeDict[str1][0]), int(self.nodeDict[str1][1])), (int(self.nodeDict[str2][0]), int(self.nodeDict[str2][1])), (0,0,255), 1)
        cv2.imwrite(self.imageLocation, self.lineVariable)
        self.textVariable = cv2.putText(self.newImage, numbering, (int((self.nodeDict[str1][0]+self.nodeDict[str2][0])/2), int((self.nodeDict[str1][1]+self.nodeDict[str2][1])/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
        cv2.imwrite(self.imageLocation, self.textVariable)


    def saveAndExit(self):
        # If we save changes, whatever we have made changes in the contour map have to be make them permanent so delete previous copy and only maintain current one 
        os.remove(self.destinationFolder + '/pointsCollection.jpg')
        sop.copy(self.imageLocation, self.destinationFolder)

        # Save changes to valley nodes by opening file in writing mode
        fileLocation = self.originalImageFileLocation + '/valleyNodes.txt'
        with open(fileLocation, 'w') as file:
            for nodeName, nodePoint in self.nodeDict.items():
                file.write(str(float(nodePoint[0])) + ' ' + str(float(nodePoint[1])) + ' ' + str(float(nodePoint[2])) + '\n')
            file.close()

        # Similarly save changes made in entire opertion for segments
        fileLocation = self.originalImageFileLocation + '/valleySegments.txt'
        with open(fileLocation, 'w') as file:
            for segmentName, segmentNodes in self.segmentDict2.items():
                file.write(segmentNodes[0][1:] + ' ' + segmentNodes[1][1:] + ' 1 1\n')
            file.close()

        self.dismissAndExit()   # Exit from the window



class ElevationValueORvalleyNode(tk.Toplevel):
    # This class is for sub-window when it comes to the elevation value input for the ridge line, we can either select a valley node as start or 
    # end point of the ridge line or provide elevation value for that selected point on the contour map

    def __init__(self, parent):
        # Set top level window on the previous defined window
        super().__init__(parent)

        # Set icon and general properties for this top-level window
        self.currentWorkingDirectory = os.getcwd()
        self.pathToIcon = self.currentWorkingDirectory + r'\guiIcon.ico'
        self.wm_iconbitmap(self.pathToIcon)
        self.minsize(300,320)
        self.maxsize(300,320)
        self.title("Elevation Or Node")

        # If window is tried to closed by cross icon, call a method rather than directly closing it
        self.protocol("WM_DELETE_WINDOW", self.onCrossSteps)
        self.operationType = parent.operationTypeVar
        self.elevationFromWindow = parent.elevationFromWindow

        # Separtor line design through canvas
        self.newCanvas = tk.Canvas(self)
        self.newCanvas.create_line(0, 160, 125, 160, dash=(1,1))
        self.newCanvas.create_line(175, 160, 300, 160, dash=(1,1))
        self.newCanvas.place(x=0,y=0,relheight=1,relwidth=1)
        # General design for the buttons
        ttk.Style().configure('general.TMenubutton', font=('Times italic bold', 13))
        ttk.Style().map('general.TMenubutton', foreground=[('pressed', 'yellow'), ('active', 'red'), ('!disabled', 'black')], background=[('pressed', 'red'), ('active', 'yellow'), ('!disabled', 'white')])
        
        # Upper Element's design related
        ttk.Label(self, text='Either provide elevation', foreground='red', font='Times 17 italic bold').place(x=150, y=30, anchor=tk.CENTER)
        ttk.Label(self, text='Elevation: ', foreground='black', font='Times 15 italic bold').place(x=145, y=65, anchor=tk.E)
        ttk.Entry(self, textvariable=self.elevationFromWindow, foreground='black', font='helvetica 13', width=10).place(x=155,y=65,anchor=tk.W)
        # self.warningLabel = ttk.Label(self, text='Provide a numerical value!', foreground='red', font='Times 13 italic bold')
        # self.warningLabel.place(x=150,y=90,anchor=tk.CENTER)
        ttk.Button(self, text='Enter', command=self.testAndProceed).place(x=150,y=120,anchor=tk.CENTER)

        # Lower Element's design related
        ttk.Label(self, text='OR', foreground='red', font='Times 17 italic bold').place(x=150, y=160, anchor=tk.CENTER)
        ttk.Label(self, text='Select nearby valley node ', foreground='red', font='Times 17 italic bold').place(x=150, y=190, anchor=tk.CENTER)
        self.Node = tk.StringVar()
        self.menuButton = ttk.Menubutton(self, text='Select valley node: ', style='general.TMenubutton')
        self.menulist = tk.Menu(self.menuButton, tearoff=0)
        self.menuButton['menu'] = self.menulist
        self.menuButton.place(x=150,y=230,anchor=tk.CENTER)
        # self.warningLabel1 = ttk.Label(self, text='Select appropriate node!', foreground='red', font='Times 13 italic bold')
        # self.warningLabel1.place(x=150,y=260,anchor=tk.CENTER)
        ttk.Button(self, text='Enter', command=self.selectFromMenu).place(x=150,y=290,anchor=tk.CENTER)

        # Add top three nearest node of the point selected in the menu list
        for ele in parent.topThreeNodes:
            newNumbering = 'N' + str(ele)
            self.menulist.add_radiobutton(label=newNumbering, value=newNumbering, variable=self.Node, command=self.updateNode)
        

    def removeLabels(self):
        # Remove labels, if exist any
        try:
            self.warningLabel.destroy()
        except:
            pass

        try:
            self.warningLabel1.destroy()
        except:
            pass
    

    def updateNode(self):
        # Remove error label and then set that selected node
        self.removeLabels()
        self.menuButton['text'] = self.Node.get()

    
    def selectFromMenu(self):
        # Before selecting final choice, remove label if exist any, And if not selected node yet give error message
        self.removeLabels()
        tempStr = self.Node.get()
        
        if tempStr == '':   # If not selected yet
            self.warningLabel1 = ttk.Label(self, text='Select appropriate node!', foreground='red', font='Times 13 italic bold')
            self.warningLabel1.place(x=150,y=260,anchor=tk.CENTER)
            return
        
        # Provide index of the valley node selected and exit from the window
        self.operationType.set(int(tempStr[1:]))
        self.quit()
        self.destroy()


    def onCrossSteps(self):
        # If either option has not selected but instread window tried to close by cross icon on top right then give error message to both 
        # option stating that you have to select either option
        self.removeLabels() # Remove any label if exist previously
        self.warningLabel = ttk.Label(self, text="Mandatory! Either proceed through this.", foreground='red', font='Times 13 italic bold')
        self.warningLabel.place(x=150,y=90,anchor=tk.CENTER)
        self.warningLabel1 = ttk.Label(self, text='Or through this!', foreground='red', font='Times 13 italic bold')
        self.warningLabel1.place(x=150,y=260,anchor=tk.CENTER)


    def testAndProceed(self):
        # If elevation value is inputed and want to proceed through this option, then corresponding error message has to be shown
        self.removeLabels()
        strVar = self.elevationFromWindow.get()

        if(strVar == ''):   # If value has not provided yet
            self.warningLabel = ttk.Label(self, text="Enter a numerical value!", foreground='red', font='Times 13 italic bold')
            self.warningLabel.place(x=150,y=90,anchor=tk.CENTER)
            return

        # Check whether provided value is either a float or even when it is float, is it a acceptable float value or not
        try:
            isFloatable = float(self.elevationFromWindow.get())
            
            if (isFloatable <= 0):  # Negative float value provided
                self.warningLabel = ttk.Label(self, text="Enter a value > 0", foreground='red', font='Times 13 italic bold')
                self.warningLabel.place(x=150,y=90,anchor=tk.CENTER)
                return
        except: # Value is not even a floatable
            self.warningLabel = ttk.Label(self, text="Elevation can't be Non-numerical!", foreground='red', font='Times 13 italic bold')
            self.warningLabel.place(x=150,y=90,anchor=tk.CENTER)
            return
        
        # If everything is ok, indicate that option selected is elevation (by setting 0 value in operationType), and exit from the window
        self.operationType.set(0)
        self.quit()
        self.destroy()



class RidgeLinesWindow(CommonSubWindowProperties, ZoomInZoomOutFunctionality):
    # After selecting option for providing the inputs for ridge lines, this window will hold all the design sapects
 
    def __init__(self, parent):
        # Set top level window on the previous defined window
        super().__init__(parent)

        # Fix new geometry for this top-level window
        self.geometry('500x320')
        self.title('Ridge Lines')

        # Line separator's design aspects
        self.newCanvas = tk.Canvas(self)
        self.newCanvas.create_line(0, 220, 500, 220, dash=(5,1))
        self.newCanvas.create_line(250, 220, 250, 320, dash=(1,1))
        self.newCanvas.place(x=0,y=0,relheight=1,relwidth=1)

        # Data in-take design
        ttk.Label(self, text="Step: Add Ridge lines one after another", foreground='black', font='Times 20 italic bold').place(x=250, y=40, anchor=tk.CENTER)
        ttk.Label(self, text="One ridge line is made of continuous nodes", foreground='red', font='Times 17 italic bold').place(x=250, y=90, anchor=tk.CENTER)
        ttk.Label(self, text='(To save, press enter or close input window)', foreground='black', font='Times 17 italic bold').place(x=250, y=120, anchor=tk.CENTER)
        ttk.Button(self, text='Add ridge line', style='generalButton.TButton', command=self.addNewLine).place(x=250,y=170,anchor=tk.CENTER)
        
        # Botton Section
        ttk.Button(self, text='Dismiss!', style='navButton.TButton', command=self.dismissAndExit).place(x=125,y=270,anchor=tk.CENTER)
        ttk.Button(self, text='Save!', style='navButton.TButton', command=self.saveAndExit).place(x=375,y=270,anchor=tk.CENTER)

        # Store all the nodes of the ridge line, ridge segments will be formed by continuous joining of these nodes in order 
        self.nodeCoordinates = []
        # Take track of only last selected node
        self.lastNodeCoordinates = []
        # Store segments in below after consideration from node coordinates
        self.ridgeSegments = []
        # Below variable is for making track whether our elevation input window is active or not
        self.runIndication = 1
        # Temporarily store all the valley nodes to consider for the starting or ending node of a ridge line
        self.valleyNodes = []
        # Top three nearest valley nodes for a point selected on the contour map will be stored in below list
        self.topThreeNodes = []
        # Get elevation value from elevation window in below variable
        self.elevationFromWindow = tk.StringVar()
        # Whether selected opertion type was valley node selection or elevation value input, would be confirmed by this variable
        self.operationTypeVar = tk.IntVar()

        # Collect all the valley nodes from the local storage
        try:

            newFileName = self.originalImageFileLocation + '/valleyNodes.txt'
            pointCloud = np.loadtxt(newFileName,skiprows=0,ndmin=2)
            nodeCoordinates = pointCloud[:,0:]
            
            for index in range(0, len(nodeCoordinates)):
                self.valleyNodes.append([nodeCoordinates[index][0], nodeCoordinates[index][1], nodeCoordinates[index][2]])

        except:
            pass

        # In order to make dismiss and save button effective in case when we want to erase out last entered ridge lines, we save work in chunks, hence 
        # in order to make this operation successful we have to maintain track of last ridge nodes selected in previous operations
        try:
            newFileName = self.originalImageFileLocation + '/ridgeNodes.txt'
            pointCloud = np.loadtxt(newFileName,skiprows=0,ndmin=2)
            nodes = pointCloud[:,0]
            self.initialCount = len(nodes)  # This will help in maintaining indexes for the next entered ridge nodes for the ridge segments

        except:
            self.initialCount = 0   # Initialise to zero means we are starting it freshly


    def mouseHandlingForNodePoints(self, event, x, y, flags, param):
        # Two opertions needed here, one is for zoomable window and second when node point is selected from the contour map

        if(event==10): # This is for zoom-in or zoom-out
            self.zoomInZoomOut(x, y, flags)
        
        elif (event == cv2.EVENT_LBUTTONDOWN or event==cv2.EVENT_RBUTTONDOWN) and self.runIndication == 1:
            # Make scale value very precise
            self.redefineScale()

            # Get actual x,y coordinate from the contour map selected point
            x_ = self.imageInitialX+(int(x*self.scaleFactorX))
            y_ = self.imageInitialY+(int(y*self.scaleFactorY))
            
            self.reconstructWindow()
            cv2.imshow('Please select node points for ridge line...', self.newImage)

            # Below code is for selecting top three nearest valley node from the point selected in the contour map
            # Below list is for maintain a 2D list where every block is for the valley node where first index will square of the distance between 
            # selected point and valley node and second index is the index of the valley node
            diffAndNum = []
            nn = 1  # Variable for index track

            for ele in self.valleyNodes:
                diffAndNum.append([(float(x_) - ele[0])*(float(x_) - ele[0]) + (float(y_) - ele[1])*(float(y_) - ele[1]), nn])
                nn = nn + 1

            diffAndNum.sort()   # Sort this list in increasing order, first three blocks represent the shortest distance valley nodes
            nn = 0
            self.topThreeNodes.clear()  # Clear previous data if exist

            for index in range(len(diffAndNum)):    # Get top three nodes if exist
                if nn>2:
                    break
                self.topThreeNodes.append(diffAndNum[index][1])
                nn = nn + 1

            self.runIndication = 0  # This indicates that we should not carry out any operation because we have to take input from sub-window for elevation
            
            self.operationTypeVar.set(-1)   # Indicates nothing is entered
            self.elevationFromWindow.set('')

            # Create elevation input window
            eVI = ElevationValueORvalleyNode(self)
            eVI.grab_set()
            eVI.mainloop()

            self.runIndication = 1  # This indicates that we have get elevation information, so we can proceed for other operations
            
            self.operationType = self.operationTypeVar.get()

            if self.operationType == 0:     # If elevation information is from the elevation value inputed
                # Mark a circle on the contour map to indicate a new selected node and Add node to list
                self.newImage = cv2.imread(self.imageLocation, 1)
                self.circleVariable = cv2.circle(self.newImage, (x_, y_), 2, (0, 0, 255), -1)
                cv2.imwrite(self.imageLocation, self.circleVariable)
                self.nodeCoordinates.append([x_, y_, float(self.elevationFromWindow.get())])

                # If there was point selected previously or new nodes before this then add segment between that point and current node
                if len(self.lastNodeCoordinates) > 0:
                    # Mark the line segment on the contour map
                    self.newImage = cv2.imread(self.imageLocation, 1)
                    self.lineVariable = cv2.line(self.newImage, (int(self.lastNodeCoordinates[0]), int(self.lastNodeCoordinates[1])), (x_, y_), (0,0,255), 1)
                    cv2.imwrite(self.imageLocation, self.lineVariable)
                    # Add the segment in the list 
                    self.ridgeSegments.append([self.lastNodeCoordinates[4], self.initialCount + len(self.nodeCoordinates), self.lastNodeCoordinates[3], 2])
                    
                # Store current node for coming opertion in order to make a line segment
                self.lastNodeCoordinates = [x_, y_, float(self.elevationFromWindow.get()), 2, self.initialCount + len(self.nodeCoordinates)]

            else:   # If we proceed through selecting a valley node
                # As stated above, If there was any node selected previously, make a line segment between that node and current node
                if len(self.lastNodeCoordinates) > 0:
                    # Here, valley node already have circle on the point but we need to show line segment joining last node and current node
                    self.newImage = cv2.imread(self.imageLocation, 1)
                    self.lineVariable = cv2.line(self.newImage, (int(self.lastNodeCoordinates[0]), int(self.lastNodeCoordinates[1])), (int(self.valleyNodes[self.operationType-1][0]), int(self.valleyNodes[self.operationType-1][1])), (0,0,255), 1)
                    cv2.imwrite(self.imageLocation, self.lineVariable)
                    # Add the segment in the list
                    self.ridgeSegments.append([self.lastNodeCoordinates[4], self.operationType, self.lastNodeCoordinates[3], 1])

                # Store current node for coming opertion in order to make a line segment
                self.lastNodeCoordinates = [int(self.valleyNodes[self.operationType-1][0]), int(self.valleyNodes[self.operationType-1][1]), int(self.valleyNodes[self.operationType-1][2]), 1, self.operationType]
        
        else:   # If mouse handling contain other than zooming or point selection, don't process it
            return
        
        # Refresh the window after making all of changes
        self.reconstructWindow()
        cv2.imshow('Please select node points for ridge line...', self.newImage)


    def addNewLine(self):
        # Prepare for fresh ridge line input
        self.lastNodeCoordinates.clear()    # If any node is stores in list because previous operation were carried out for last ridge line inputed, erase that node

        # Present the contour map in the openCV window
        self.newImage = cv2.imread(self.imageLocation, 1)
        self.setImageWindowDimensions()
        self.newImage = cv2.resize(self.newImage,self.resizedDimensions)
        cv2.imshow('Please select node points for ridge line...', self.newImage)
        cv2.setMouseCallback('Please select node points for ridge line...', self.mouseHandlingForNodePoints)
        
        # Above created window should not get closed by entering any key of the keyboard but "Enter" and "Cross-icon"
        while(True):
            newKey = cv2.waitKey(0)
            if(self.runIndication and (newKey == -1 or newKey == 13)):
                break

        cv2.destroyAllWindows()


    def saveAndExit(self):
        # In order to make all the changes we have marked on the contour map, We should show modified image for all next operations
        os.remove(self.destinationFolder+'/pointsCollection.jpg')
        sop.copy(self.imageLocation, self.destinationFolder)

        # Save the ridge nodes in the local storage
        fileLocation = self.originalImageFileLocation + '/ridgeNodes.txt'
        with open(fileLocation, 'a') as file:   # File opened in 'append' mode
            newFileName = self.originalImageFileLocation + '/ridgeNodes.txt'
            pointCloud = np.loadtxt(newFileName,skiprows=0,ndmin=2)
            nodes = pointCloud[:,0]
            lastCounted = len(nodes)

            for items in self.nodeCoordinates:
                lastCounted = lastCounted + 1
                file.write(str(items[0]) + ' ' + str(items[1]) + ' ' + str(items[2]) + '\n')
            
            file.close()

        # Save the ridge segments in the local storage
        fileLocation = self.originalImageFileLocation + '/ridgeSegments.txt'
        with open(fileLocation, 'a') as file:
            newFileName = self.originalImageFileLocation + '/ridgeSegments.txt'
            pointCloud = np.loadtxt(newFileName,skiprows=0,ndmin=2)
            nodes = pointCloud[:,0]
            lastCounted = len(nodes)

            for items in self.ridgeSegments:
                lastCounted = lastCounted + 1
                file.write(str(items[0]) + ' ' + str(items[1]) + ' ' + str(items[2]) + ' ' + str(items[3]) + '\n')
            
            file.close()

        # Finally dismiss the window and exit from it
        self.dismissAndExit()



class ContourPointsWindow(CommonSubWindowProperties, ZoomInZoomOutFunctionality):
    # In this window, we have functionality: Points are selected on the contour line of the contour map, then elevation value provided to these selected points
    
    def __init__(self, parent):
        # Set top level window on the previous defined window
        super().__init__(parent)

        # Fix new geometry for this top-level window
        self.geometry('500x300')
        self.title('Fix elevation contour')

        # Line Separator's design
        self.newCanvas = tk.Canvas(self)
        self.newCanvas.create_line(0, 220, 500, 220, dash=(5,1))
        # self.newCanvas.create_line(250, 220, 250, 300, dash=(1,1))
        self.newCanvas.place(x=0,y=0,relheight=1,relwidth=1)

        # Design part for functionality
        ttk.Label(self, text='Step: Select points over same elevation contour', foreground='black', font='Times 17 italic bold').place(x=250, y=40, anchor=tk.CENTER)
        ttk.Button(self, text='Collect Points', style='generalButton.TButton', command=self.selectPointsOnContour).place(x=125,y=90,anchor=tk.CENTER)
        ttk.Label(self, text='Elevation: ', foreground='black', font='Times 15 italic bold').place(x=250, y=90, anchor=tk.W)
        self.elevationValue = tk.StringVar()
        ttk.Entry(self, textvariable=self.elevationValue, foreground='black', font='helvetica 13', width=10).place(x=400,y=90,anchor=tk.CENTER)
        # self.nodeAddWarning = ttk.Label(self, text='test label !!!', foreground='red', font='Times 13 italic bold')
        # self.nodeAddWarning.place(x=250,y=120,anchor=tk.CENTER)
        ttk.Button(self, text='Add Contour', style='intermediateButton.TButton', command=self.addNewContour).place(x=250,y=170,anchor=tk.CENTER)
        
        # Bottom Section
        ttk.Button(self, text='Dismiss!', style='navButton.TButton', command=self.dismissAndExit).place(x=125,y=260,anchor=tk.CENTER)
        ttk.Button(self, text='Save!', style='navButton.TButton', command=self.saveAndExit).place(x=375,y=260,anchor=tk.CENTER)

        self.initialPointsCollection = []  # All points on a contour line would be initially stored in this list
        self.finalPoints = []   # After process the initial points
        self.segmentsNodes = [] # For new segments created
        
        # In order to give node indexing for the newly formed segments, we have to make count of prviously existing nodes
        try:
            newFileName = self.originalImageFileLocation + '/contourNodes.txt'
            pointCloud = np.loadtxt(newFileName,skiprows=0,ndmin=2)
            nodes = pointCloud[:,0]
            self.currentNumber = len(nodes) # Note down count
        except:
            self.currentNumber = 0  # If there doesn't exist previous file then it is a fresh starting


    def mouseHandlingForContourPoints(self, event, x, y, flags, param):
        # For zoom-in or zoom-out
        if(event==10):
            self.zoomInZoomOut(x, y, flags)
        # On left or right click by mouse, that point should be collected
        elif event == cv2.EVENT_LBUTTONDOWN or event==cv2.EVENT_RBUTTONDOWN:
            self.redefineScale()    # Choose current precise scale
            ll = []
            ll.append(self.imageInitialX+(int(x*self.scaleFactorX)))
            ll.append(self.imageInitialY+(int(y*self.scaleFactorY)))
            self.initialPointsCollection.append(ll) # Store nodes in initial list
            # Mark a tiny circle on the contour map to show location of node selected
            self.newImage = cv2.imread(self.imageLocation, 1)
            self.circleVariable = cv2.circle(self.newImage, (self.imageInitialX+(int(x*self.scaleFactorX)), self.imageInitialY+(int(y*self.scaleFactorY))), 2, (0,255,0), 2)
            cv2.imwrite(self.imageLocation, self.circleVariable)
        # If not hitted of above two, escape from below procedure
        else:
            return
        
        # Refresh the openCV window
        self.reconstructWindow()
        cv2.imshow('Select points on contour line...', self.newImage)


    def selectPointsOnContour(self):
        # Create openCV window and show contour map in that
        self.newImage = cv2.imread(self.imageLocation, 1)
        self.setImageWindowDimensions()

        # Erase-out any previosly entered nodes which were selected for last contour
        self.initialPointsCollection.clear()

        self.newImage = cv2.resize(self.newImage,self.resizedDimensions)
        cv2.imshow('Select points on contour line...', self.newImage)
        cv2.setMouseCallback('Select points on contour line...', self.mouseHandlingForContourPoints)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def contourPointsErrorHandling(self, newLabel):
        # Remove label, if exist any
        try:
            self.nodeAddWarning.destroy()
        except:
            pass
        
        # Add new label if required
        if newLabel != 'None':
            self.nodeAddWarning = ttk.Label(self, text=newLabel, foreground='red', font='helvetica 13')
            self.nodeAddWarning.place(x=250,y=120,anchor=tk.CENTER)


    def addNewContour(self):
        # Remove error handling label if exist
        labelToShow = 'None'
        self.contourPointsErrorHandling(labelToShow)

        # Check if any error is there with elevation value provided and no of points selected
        try:
            # If value is conversionable to float then we are good to go
            self.elevation = float(self.elevationValue.get())

            # If there has not been selected any contour nodes yet 
            if len(self.initialPointsCollection) == 0:
                labelToShow = 'Select points on contour line'
                self.contourPointsErrorHandling(labelToShow)
                return

        except:
            # If value entered is not float show corresponding error and put flag to reset it
            labelToShow = 'Enter a float value for elevation'
            self.contourPointsErrorHandling(labelToShow)
            return
        
        elevation = float(self.elevationValue.get())    # Get elevation value from input field
        
        # If everything is good, Iterate over all selected nodes and save these nodes and form appropriate segments
        for i in range(0, len(self.initialPointsCollection)):

            if i>0:     # Segments only will be formed if there are more than one nodes points are there
                ll = []
                ll.append(self.currentNumber)
                ll.append(self.currentNumber+1)
                self.segmentsNodes.append(ll)   # Segments is made-up of previously entered node and current node to be inserted
                
                # Show corresponding segment on the contour map
                self.newImage = cv2.imread(self.imageLocation, 1)
                self.lineVariable = cv2.line(self.newImage, (self.initialPointsCollection[i-1][0], self.initialPointsCollection[i-1][1]), (self.initialPointsCollection[i][0], self.initialPointsCollection[i][1]), (0,0,255), 1)
                cv2.imwrite(self.imageLocation, self.lineVariable)
            
            self.currentNumber = self.currentNumber + 1     # Increment pointer for count
            self.initialPointsCollection[i].append(elevation)       # Elevation value has to be append with x,y coordinates

            self.finalPoints.append(self.initialPointsCollection[i])        # Now final turn is to set this node into contour node list

        # Erase out data from initial list after recorded them
        self.initialPointsCollection.clear()
        self.elevationValue.set('')


    def saveAndExit(self):
        # In order to make all the changes we have marked on the contour map, We should show modified image for all next operations
        os.remove(self.destinationFolder + '/pointsCollection.jpg')
        sop.copy(self.imageLocation, self.destinationFolder)

        # Save contour nodes on the local storage
        fileLocation = self.originalImageFileLocation + '/contourNodes.txt'
        with open(fileLocation, 'a') as file:
            for items in self.finalPoints:
                file.write(str(items[0]) + ' ' + str(items[1]) + ' ' + str(items[2]) + '\n')
            file.close()

        # Store contour segments on the local storage 
        fileLocation = self.originalImageFileLocation + '/contourSegments.txt'
        with open(fileLocation, 'a') as file:
            for items in self.segmentsNodes:
                file.write(str(items[0]) + ' ' + str(items[1]) + ' 3 3\n')
            file.close()
        
        # Finally dismiss the window and exit from it
        self.dismissAndExit()



class PointsCollectionWindow(tk.Tk, ZoomInZoomOutFunctionality):
    # After selecting the scale factor, we have to get basic geometry of the catchment area, for that we have four main inputs: Catchment boundry, 
    # valley lines, ridge lines And contour lines. When these four will combine, we will have a full geometry of the catchment area.
    # Along with just landing page for above four main category, We also defined catchment boundary collection openCV window design in this class

    def __init__(self, imageAddress, scaleMultiple):
        # Set top level window on the previous defined window
        super().__init__()

        # Get main directory where all required files are present 
        self.appWorkingDirectory = os.getcwd()
        self.pathToIcon = self.appWorkingDirectory + r'\guiIcon.ico'
        self.wm_iconbitmap(self.pathToIcon)

        # Fix new geometry for this window
        self.geometry('850x550')
        self.title('Step by Step data from Contour Map')

        # Save previous class data
        self.originalImageAddress = imageAddress
        self.imageLocation = self.originalImageAddress
        self.scaleMultiple = scaleMultiple

        # separate folder name from file address
        i,j = 0,0
        while i < len(self.originalImageAddress):
            if self.originalImageAddress[i] == '/':
                j = i
            i+=1
        j+=1
        # Get image name and its location
        self.originalImageFileLocation = self.originalImageAddress[:j-1]
        self.originalImageName = self.originalImageAddress[j:]

        # Getting background image object
        self.backgroundImageLocation = self.appWorkingDirectory + '\i1.png.'
        self.backgroundImageObject = tk.PhotoImage(file = self.backgroundImageLocation)

        # All properties for buttons
        ttk.Style().configure('TButton', padding=2, relief="flat")
        ttk.Style().configure('generalButton.TButton', font=('Times italic bold', 13))
        ttk.Style().map('generalButton.TButton', foreground=[('pressed', 'blue'), ('active', 'red'), ('!disabled', 'red')], background=[('pressed', 'yellow'), ('active', 'black'), ('!disabled', 'yellow')])
        ttk.Style().configure('intermediateButton.TButton', font=('Times italic bold', 15))
        ttk.Style().map('intermediateButton.TButton', foreground=[('pressed', 'blue'), ('active', 'black'), ('!disabled', 'black')], background=[('pressed', 'red'), ('active', 'black'), ('!disabled', 'black')])
        ttk.Style().configure('navButton.TButton', font=('Times italic bold', 18))
        ttk.Style().map('navButton.TButton', foreground=[('pressed', 'blue'), ('active', 'black'), ('!disabled', 'black')], background=[('pressed', 'red'), ('active', 'black'), ('!disabled', 'black')])

        # Background setup
        self.newCanvas = tk.Canvas(self)
        self.newCanvas.create_image(0,0,anchor=tk.NW,image=self.backgroundImageObject)
        self.newCanvas.create_line(425, 40, 425, 400)
        self.newCanvas.create_line(50, 400, 800, 400)
        self.newCanvas.place(x=0,y=0,relheight=1,relwidth=1)

        # Section 1
        ttk.Label(self, text='Step 1: Decide on Catchment inclusion', foreground='black', font='Times 17 italic bold').place(x=212, y=60, anchor=tk.CENTER)
        ttk.Label(self, text='(To save, press enter or close input window)', foreground='black', font='Times 17 italic bold').place(x=212, y=90, anchor=tk.CENTER)
        ttk.Button(self, text='Mark Catchment Boundry', style='generalButton.TButton', command=self.drawCatchment).place(x=212,y=140,anchor=tk.CENTER)
        
        # Section 2
        ttk.Label(self, text='Step 2: Draw valley lines', foreground='black', font='Times 17 italic bold').place(x=212, y=260, anchor=tk.CENTER)
        ttk.Label(self, text='(Valley lines must be connected)', foreground='black', font='Times 17 italic bold').place(x=212, y=290, anchor=tk.CENTER)
        ttk.Button(self, text='Figure out valley lines', style='generalButton.TButton', command=self.selectValleyPoints).place(x=212,y=340,anchor=tk.CENTER)
        
        # Section 3
        ttk.Label(self, text='Step 3: Draw ridge lines', foreground='black', font='Times 17 italic bold').place(x=638, y=60, anchor=tk.CENTER)
        ttk.Label(self, text='(May not be connected)', foreground='black', font='Times 17 italic bold').place(x=638, y=90, anchor=tk.CENTER)
        ttk.Button(self, text='Figure out ridge lines', style='generalButton.TButton', command=self.selectRidgePoints).place(x=638,y=140,anchor=tk.CENTER)
        
        # Section 4
        ttk.Label(self, text='Step 4: Select Contour Points', foreground='black', font='Times 17 italic bold').place(x=638, y=260, anchor=tk.CENTER)
        ttk.Label(self, text='(Select points corresponding to an elevation)', foreground='black', font='Times 17 italic bold').place(x=638, y=290, anchor=tk.CENTER)
        ttk.Button(self, text='Add contour points', style='generalButton.TButton', command=self.selectContourPoints).place(x=638,y=340,anchor=tk.CENTER)
        
        # Section 5
        ttk.Button(self, text='Dismiss!', style='navButton.TButton', command=self.destroyCurrentWindow).place(x=150,y=450,anchor=tk.W)
        ttk.Button(self, text='Proceed Next!', style='navButton.TButton', command=self.saveAndClose).place(x=700,y=450,anchor=tk.E)

        # Catchment Boundry points related
        self.lastCatchmentPoint = []        # Just previous point than current in case it exist, will be helpful for segment creation
        self.firstCatchmentPoint = []       # Very first point selected on the contour map for cathcment boundary, this will help for completion of catchment loop
        self.cntNum = 0                     # count variable
        # Below variable will helpful for making track of one by one entry of catchment node along with the elevation value, we should not proceed forward 
        # without being provided the elevation value for previous node selected
        self.runIndication = 1
        self.catchmentNodes = []            # List for the catchment nodes
        self.catchmentSegments = []         # List for the catchment segments

    
    def mouseHandlingForCatchmentPoints(self, event, x, y, flags, param):
        # In majority of cases, we will require the mouse handling for the zoomable window properties and left or right click functionality.

        if(event==10): # This is for zoom-in or zoom-out
            self.zoomInZoomOut(x, y, flags)
        
        elif (event == cv2.EVENT_LBUTTONDOWN or event==cv2.EVENT_RBUTTONDOWN) and self.runIndication == 1:
            # On left or right click by mouse, that point should be collected
            self.redefineScale()
            self.newImage = cv2.imread(self.imageLocation, 1)

            # Save the first catchment node, this will be helpful for completion of catchment loop and hence the final segment of the catchment boundry
            if len(self.firstCatchmentPoint) == 0:
                self.firstCatchmentPoint.append(self.imageInitialX+(int(x*self.scaleFactorX)))
                self.firstCatchmentPoint.append(self.imageInitialY+(int(y*self.scaleFactorY)))
            
            # If there exist any previously selected node, make a line segment between current point and last point
            if len(self.lastCatchmentPoint) > 0:
                self.lineVariable = cv2.line(self.newImage, (self.lastCatchmentPoint[0], self.lastCatchmentPoint[1]), (self.imageInitialX+(int(x*self.scaleFactorX)), self.imageInitialY+(int(y*self.scaleFactorY))), (0,0,255), 1)
                cv2.imwrite(self.imageLocation, self.lineVariable)
                self.lastCatchmentPoint.clear() # Erase-out last node

            # Make last node for next operation
            self.lastCatchmentPoint.append(self.imageInitialX+(int(x*self.scaleFactorX)))
            self.lastCatchmentPoint.append(self.imageInitialY+(int(y*self.scaleFactorY)))

            # Mark a circle around the selected point
            self.circleVariable = cv2.circle(self.newImage, (self.imageInitialX+(int(x*self.scaleFactorX)), self.imageInitialY+(int(y*self.scaleFactorY))), 2, (0,255,0), 2)
            cv2.imwrite(self.imageLocation, self.circleVariable)

            self.reconstructWindow()
            cv2.imshow('Please select boundry points...', self.newImage)

            # Get the elevation value from the new sub-window created
            self.runIndication = 0  # This indicates that we should not run any other functionality except getting elevation value from sub-window
            self.elevationFromWindow = tk.StringVar()   # String variable for elevation value storage
            eVI = ElevationValueInputFromWindow(self)
            eVI.mainloop()
            self.runIndication = 1  # Now making permission for having any other activity

            self.lastCatchmentPoint.append(float(self.elevationFromWindow.get()))   # Get z-coordinate from appending elevation value
            self.catchmentNodes.append(copy.deepcopy(self.lastCatchmentPoint))  # Fianlly save the current catchment node

            return
        
        else:
            # On using any other mouse functionality, it should have no effect on openCV window
            return
        
        self.reconstructWindow()
        cv2.imshow('Please select boundry points...', self.newImage)


    def drawCatchment(self):
        # Before starting anything, all previous record should be vanished
        self.lastCatchmentPoint.clear()
        self.firstCatchmentPoint.clear()

        # Load image for the openCV window
        self.newImage = cv2.imread(self.imageLocation, 1)
        self.setImageWindowDimensions()

        self.newImage = cv2.resize(self.newImage,self.resizedDimensions)
        cv2.imshow('Please select boundry points...', self.newImage)
        cv2.setMouseCallback('Please select boundry points...', self.mouseHandlingForCatchmentPoints)
        
        # Our openCV window should only closed by the pressing 'Enter' or using 'Cross-icon' and should not be closed during elevation value input
        while(True):
            newKey = cv2.waitKey(0)
            if(self.runIndication and (newKey == -1 or newKey == 13)):
                break
        cv2.destroyAllWindows()

        # If we have selected catchment nodes, make sure they get saved on local storage
        if len(self.firstCatchmentPoint)>0:
            self.redefineScale()
            # Create final segment for closing the loop for the catchment boundry
            self.newImage = cv2.imread(self.imageLocation, 1)
            self.lineVariable = cv2.line(self.newImage, (self.lastCatchmentPoint[0], self.lastCatchmentPoint[1]), (self.firstCatchmentPoint[0], self.firstCatchmentPoint[1]), (0,0,255), 1)
            cv2.imwrite(self.imageLocation, self.lineVariable)
            
            # Save catchment nodes on the local storage
            fileLocation = self.originalImageFileLocation + '/catchmentNodes.txt'
            with open(fileLocation, 'w') as file:
                for items in self.catchmentNodes:
                    file.write(str(items[0]) + ' ' + str(items[1]) + ' ' + str(items[2]) + '\n')
                file.close()

            # Save catchment segments on the local storage
            fileLocation = self.originalImageFileLocation + '/catchmentSegments.txt'
            with open(fileLocation, 'w') as file:
                totSize = len(self.catchmentNodes)
                for i in range(totSize-1):
                    file.write(str(i+1) + ' ' + str(i+2) + ' 0 0\n')
                if totSize>2:
                    file.write(str(1) + ' ' + str(totSize) + ' 0 0\n')
                file.close()


    def resetCondition(self):
        # In order to make the appearance like we have made changes on clicking 'Save' And erased all changes on clicking 'Dismiss', contour image 
        # is copy from created folder for same. This folder will be updated based on the 'Save' Or 'Dismiss' button used
        try:
            os.remove(self.originalImageFileLocation + '/pointsCollection.jpg')
            sop.copy(self.originalImageFileLocation + '/nocombosvposrp/pointsCollection.jpg', self.originalImageFileLocation)
            sop.rmtree(self.originalImageFileLocation + '/nocombosvposrp')
        except:
            # In case some error happend, this will bypass the process
            pass

    def selectValleyPoints(self):
        # Create new window and take all inputs for valley lines
        sVP = ValleyLinesWindow(self)
        sVP.grab_set()  # This functionality transfer mobility to the only new created window (It is sVP). And we can't use button or any other functionality in the root window 
        sVP.mainloop()
        self.resetCondition()


    def selectRidgePoints(self):
        # Create new window and take all inputs for ridge lines
        sRP = RidgeLinesWindow(self)
        sRP.grab_set()
        sRP.mainloop()
        self.resetCondition()


    def selectContourPoints(self):
        # Create new window and take all inputs for contour lines
        sCP = ContourPointsWindow(self)
        sCP.grab_set()
        sCP.mainloop()
        self.resetCondition()


    def destroyCurrentWindow(self):
        self.destroy()  # Exit from current window


    def saveAndClose(self):
        # Transfer scale multiple and contour image location to next window and destroy current window
        imageLoc = self.originalImageFileLocation
        scaleMultiple = self.scaleMultiple
        self.destroy()
        nextPage = VisualiseCatchmentAreaAndProceedForFinal(imageLoc, scaleMultiple)
        nextPage.mainloop()



class InitialInformationCollectionWindow(tk.Tk, ZoomInZoomOutFunctionality):
    # This is basic window, where contour image will be selected from the local storage, Scale points and corresponding scale value will be taken as input.

    def __init__(self):
        # Make a new fresh root window
        super().__init__()
        
        # General window properties
        self.title('Basic information')
        self.geometry('800x550')
        self.minsize(850,550)
        self.maxsize(850,550)

        # All properties of buttons
        ttk.Style().configure('TButton', padding=2, relief="flat")
        ttk.Style().configure('firstPageButton.TButton', font=('Times italic bold', 13))
        ttk.Style().map('firstPageButton.TButton', foreground=[('pressed', 'blue'), ('active', 'red'), ('!disabled', 'red')], background=[('pressed', 'yellow'), ('active', 'black'), ('!disabled', 'yellow')])
        ttk.Style().configure('navButton.TButton', font=('Times italic bold', 18))
        ttk.Style().map('navButton.TButton', foreground=[('pressed', 'blue'), ('active', 'black'), ('!disabled', 'black')], background=[('pressed', 'red'), ('active', 'black'), ('!disabled', 'black')])

        # Track current working directory and form new image object for background
        self.currentWorkingDirectory = os.getcwd()
        self.imageFileLocation = self.currentWorkingDirectory + '\i1.png.'
        self.backgroundImageObject = tk.PhotoImage(file = self.imageFileLocation)
        self.pathToIcon = self.currentWorkingDirectory + r'\guiIcon.ico'
        self.wm_iconbitmap(self.pathToIcon)

        # Basic background design
        self.newCanvas = tk.Canvas(self)
        self.newCanvas.create_image(0,0,anchor=tk.NW,image=self.backgroundImageObject)
        self.newCanvas.create_line(425, 40, 425, 360)
        self.newCanvas.create_line(50, 360, 800, 360)
        self.newCanvas.place(x=0,y=0,relheight=1,relwidth=1)

        # First section start (For selecting a image for inputs):
        # Heading label
        ttk.Label(self, text='Step 1: Choose the contour-map image', foreground='black', font='Times 17 italic bold').place(x=212, y=60, anchor=tk.CENTER)
        # Button for file opening
        ttk.Button(self, text='Open', style='firstPageButton.TButton', command=self.openFile).place(x=212,y=100,anchor=tk.CENTER)
        # # Error Testing purpose only
        # self.imageIndicator = ttk.Label(self, text='testing label !!!', foreground='red', font='helvetica 13')  # self.imageIndicator.place(x=212, y=140, anchor=tk.CENTER)

        # Second section start (For scale setting of image):
        # Heading Label
        ttk.Label(self, text='Step 2: Select scale points', foreground='black', font='Times 17 italic bold').place(x=212, y=230, anchor=tk.CENTER)
        # Button to open an image with new screen and then will allow at max two points to select
        ttk.Button(self, text='Define', style='firstPageButton.TButton', command=self.setScale).place(x=212,y=270,anchor=tk.CENTER)
        # # Error Testing purpose only
        # self.scaleSetLabel = ttk.Label(self, text='testing label !!!', foreground='red', font='helvetica 13')  # self.scaleSetLabel.place(x=212, y=310, anchor=tk.CENTER)

        # Third section start (For setting scale equivalent units)
        # Below is string type variable
        self.equivalentValue = tk.StringVar()
        # Heading Label
        ttk.Label(self, text=' Step 3: Enter equivalent distance ', foreground='black', font='Times 17 italic bold').place(x=637, y=120, anchor=tk.CENTER)
        ttk.Label(self, text=' of scale segment ', foreground='black', font='Times 17 italic bold').place(x=637, y=150, anchor=tk.CENTER)
        ttk.Entry(self, textvariable=self.equivalentValue, foreground='black', font='helvetica 13').place(x=637,y=190,anchor=tk.CENTER) # For value intake
        # Buttom to check the validity and final save
        ttk.Button(self, text='Set', style='firstPageButton.TButton', command=self.setUnitValue).place(x=637,y=230,anchor=tk.CENTER)
        # # Error Testing purpose only
        # self.scaleUnitWarning = ttk.Label(self, text='testing label !!!', foreground='red', font='helvetica 13')  # self.scaleUnitWarning.place(x=637, y=270, anchor=tk.CENTER)

        # Last two buttons for the exit from application and the next page mover
        ttk.Button(self, text='Exit!', style='navButton.TButton', command=self.destroyRoot).place(x=150,y=450,anchor=tk.W)
        ttk.Button(self, text='Next!', style='navButton.TButton', command=self.nextPage).place(x=700,y=450,anchor=tk.E)

        # Variable for Error Handling part
        self.scaleError = True
        self.scaleUnit = 0
        self.scaleUnitError = True
        self.scalePointContainer = []


    def imageSelectionErrorHandling(self, newLabel):
        # Remove any previous Label if exist
        try:
            self.imageIndicator.destroy()
        except:
            pass
        # Showing corresponding Label if we want to show it
        if(newLabel != 'None'):
            self.imageIndicator = ttk.Label(self, text=newLabel, foreground='red', font='helvetica 13')
            self.imageIndicator.place(x=212, y=140, anchor=tk.CENTER)
            # Make flag to make sure that if image is reset again then we must again do whole process
            self.scaleError = True


    def scalePointsErrorHandling(self, newLabel):
        # Remove any previous Label if exist
        try:
            self.scaleSetLabel.destroy()
        except:
            pass
        # Showing corresponding Label if we want to show it
        if(newLabel != 'None'):
            self.scaleSetLabel = ttk.Label(self, text=newLabel, foreground='red', font='helvetica 13')
            self.scaleSetLabel.place(x=212, y=310, anchor=tk.CENTER)
            self.scaleError = True  # Again select the value
        else:
            newLabel = 'Kudos! scale has been decided.'
            self.scaleSetLabel = ttk.Label(self, text=newLabel, foreground='red', font='helvetica 13')
            self.scaleSetLabel.place(x=212, y=310, anchor=tk.CENTER)
            self.scaleError = False  # Value is selected


    def scaleUnitErrorHandling(self, newLabel):
        # Remove any previous Label if exist
        try:
            self.scaleUnitWarning.destroy()
        except:
            pass
        # Showing corresponding Label if we want to show it
        if(newLabel != 'None'):
            self.scaleUnitWarning = ttk.Label(self, text=newLabel, foreground='red', font='helvetica 13')
            self.scaleUnitWarning.place(x=637, y=270, anchor=tk.CENTER)
            self.scaleUnitError = True  # Again select the value
        else:
            self.scaleUnitError = False  # Value is selected
                

    def setUnitValue(self):
        # Handle cases
        try:
            # If value is conversionable to float then we are good to go
            self.scaleUnit = float(self.equivalentValue.get())
            
            if self.scaleUnit <= 0: # Value can't be negative or zero in any case
                labelToShow = 'Please select a value greater than zero.'
                self.scaleUnitErrorHandling(labelToShow)
            else:  # Remove flag because we are good to go
                labelToShow = 'None'
                self.scaleUnitErrorHandling(labelToShow)

        except:
            # If value entered is not float show corresponding error and put flag to reset it
            labelToShow = 'Please select a float value.'
            self.scaleUnitErrorHandling(labelToShow)


    def openFile(self):
        # Select the image's full path in string format
        self.imageLocation = fd.askopenfilename(title = "Select your file...", filetypes = [("Only *.png","*.png"),("Only *.jpg","*.jpg")])

        if self.imageLocation == '':  # If we closed window without select a file
            labelToShow = 'Please select image before proceed next.'
            self.imageSelectionErrorHandling(labelToShow)

        else:  # If selected then show in Label
            # get file name to show
            i,j = 0,0
            while i < len(self.imageLocation):
                if self.imageLocation[i] == '/':
                    j = i
                i+=1
            j+=1
            
            labelToShow = self.imageLocation[j:]
            self.imageSelectionErrorHandling(labelToShow)


    def mouseHandlingForScale(self, event, x, y, flags, param):
        if(event==10):  # This is for zoom-in or zoom-out
            self.zoomInZoomOut(x, y, flags)

        elif ((event == cv2.EVENT_LBUTTONDOWN or event==cv2.EVENT_RBUTTONDOWN) and (self.countPoints < 2)):
            # On left or right click by mouse, If we already selected two points we are not going to select thrid point
            self.countPoints = self.countPoints + 1 # Increment selected point's pointer counter
            self.redefineScale()
            ll = []
            ll.append(self.imageInitialX+x*self.scaleFactorX)
            ll.append(self.imageInitialY+y*self.scaleFactorY)
            self.scalePointContainer.append(ll)
            self.newImage = cv2.imread(self.imageLocation, 1)
            # Below to show a solid circle on the selected point
            self.circleVariable = cv2.circle(self.newImage, (self.imageInitialX+(int(x*self.scaleFactorX)), self.imageInitialY+(int(y*self.scaleFactorY))), 2, (0,0,255), 2)
            cv2.imwrite(self.imageLocation, self.circleVariable)
            # If we already selected two points then close the window
            if self.countPoints == 2:
                cv2.destroyWindow('Please select two points...')
                return
        
        else:
            return
        
        self.reconstructWindow()
        cv2.imshow('Please select two points...', self.newImage)


    def makeCopyOfOriginalFile(self):
        # Separate the file folder from image name
        i,j = 0,0
        while i < len(self.imageLocation):
            if self.imageLocation[i] == '/':
                j = i
            i+=1
        j+=1
            
        self.originalImageFileLocation = self.imageLocation[:j-1]
        self.originalImageName = self.imageLocation[j:]

        # make track of app's directory
        self.appWorkingDirectory = os.getcwd()

        # Operations would be like below:
        # make new temporary folder, copy that image there, rename there, make copy back to original folder and remove temporary folder
        os.chdir(self.originalImageFileLocation)
        os.mkdir('temporaryCopyPasteAndFlushFolder')
        os.chdir('temporaryCopyPasteAndFlushFolder')
        fakeDestinationFolder = os.getcwd()
        sop.copy(self.imageLocation, fakeDestinationFolder)
        os.rename(self.originalImageName, 'pointsCollection.jpg')
        sop.copy(fakeDestinationFolder+'/pointsCollection.jpg', self.originalImageFileLocation)
        os.chdir(self.originalImageFileLocation)
        sop.rmtree(fakeDestinationFolder)
        self.imageLocation = self.originalImageFileLocation + '/pointsCollection.jpg'
        os.chdir(self.appWorkingDirectory)


    def removeExtraCopy(self):
        # Remove the duplicate copy and rename variable
        os.remove(self.originalImageFileLocation + '/pointsCollection.jpg')
        self.imageLocation = self.originalImageFileLocation + '/' + self.originalImageName 


    def setScale(self):
        # If image has not been selected then go for it first
        if hasattr(self, 'imageLocation') == False or self.imageLocation is None or self.imageLocation == '':
            newLabelText = 'Please select image first.'
            self.imageSelectionErrorHandling(newLabelText)

        else:
            self.countPoints = 0  # Make sure that only two points are selected for the scale measurement 
            self.scalePointContainer.clear() # Redefine list

            self.makeCopyOfOriginalFile() # Duplicate the original image file

            self.newImage = cv2.imread(self.imageLocation, 1) # Read an image using openCV module

            self.setImageWindowDimensions()  # Fixing the size of image to be shown on the screen

            self.newImage = cv2.resize(self.newImage,self.resizedDimensions)  # Set as per above fixed sized window declaration
            cv2.imshow('Please select two points...', self.newImage) # Showing it on window
            cv2.setMouseCallback('Please select two points...', self.mouseHandlingForScale) # Set a mouse callback
            cv2.waitKey(0) # wait till operation ends
            cv2.destroyAllWindows() # Remove image window if we already got our points

            self.removeExtraCopy() # Remove the duplicate copy and reset image variable
            
            if self.countPoints < 2: # There must be exactly two points
                labelToShow = 'Please select 2 points.'
                self.scalePointsErrorHandling(labelToShow)

            elif self.scalePointContainer[0][0] == self.scalePointContainer[1][0] and self.scalePointContainer[0][1] == self.scalePointContainer[1][1]:
                # We must have two different points for scale
                labelToShow = 'Please select two distinct points.'
                self.scalePointsErrorHandling(labelToShow)
            
            else:
                # we are good to go for next operation
                labelToShow = 'None'
                self.scalePointsErrorHandling(labelToShow)   


    def flagCheck(self):

        # Checking agaist all flags to make sure no error is there
        if hasattr(self, 'imageLocation') == False or self.imageLocation is None or self.imageLocation == '':
            # If previously no image is selected or window closed without selection
            labelToShow = 'Please select image first.'
            self.imageSelectionErrorHandling(labelToShow)
            return 1

        elif self.scaleError:
            # If scale points not selected in new selected image
            labelToShow = 'Please select scale points here.'
            self.scalePointsErrorHandling(labelToShow)
            return 1

        elif self.scaleUnitError:
            # If unit value is not set yet
            labelToShow = 'Please set corresponding scale unit.'
            self.scaleUnitErrorHandling(labelToShow)
            return 1
        
        return 0


    def destroyRoot(self):
        # Destroy current window to stop program execution
        self.destroy()


    def nextPage(self):
        # Checking agaist all flags to make sure no error is there
        if (self.flagCheck() == 0):
            # Scale multiplication to be send to the next page class
            scaleMultiple = (self.scaleUnit*self.scaleUnit)/((self.scalePointContainer[0][0]-self.scalePointContainer[1][0])*(self.scalePointContainer[0][0]-self.scalePointContainer[1][0]) + (self.scalePointContainer[0][1]-self.scalePointContainer[1][1])*(self.scalePointContainer[0][1]-self.scalePointContainer[1][1]))
            scaleMultiple = math.sqrt(scaleMultiple)

            self.makeCopyOfOriginalFile() # Make duplicate copy

            imageAddress = self.imageLocation
            # Destroy current window
            self.destroy()
            # Calling the next page class
            nextWindow = PointsCollectionWindow(imageAddress, scaleMultiple)
            nextWindow.mainloop()

            os.remove(imageAddress) # Remove duplicated copy



class App(tk.Tk):
    # This class is inherited from the tk.Tk class to create landing root window for our simulator

    def __init__(self):
        # Create root window for this GUI
        super().__init__()

        # General window properties
        self.title('Hydrograph Simulator')
        self.geometry('850x550')
        # Fix the window size
        self.minsize(850,550)
        self.maxsize(850,550)

        # Get CWD for setting of the background images
        self.currentWorkingDirectory = os.getcwd()
        self.imageFileLocation = self.currentWorkingDirectory + '\i1.png'
        self.pathToIcon = self.currentWorkingDirectory + r'\guiIcon.ico'

        # Set icon for the gui
        self.wm_iconbitmap(self.pathToIcon)
        # Create an image object for background part and then place it
        # Note that images will be only accepted in png, ppm etc but not in jpg
        self.backgroundImageObject = tk.PhotoImage(file = self.imageFileLocation)
        ttk.Label(self, image=self.backgroundImageObject).place(x=0, y=0, relheight=1, relwidth=1)

        # Define ttk common styles for the button
        ttk.Style().configure('TButton', padding=10, relief="flat")
        ttk.Style().configure('landPageButton.TButton', font=('Times italic bold', 15))
        ttk.Style().map('landPageButton.TButton', foreground=[('pressed', 'blue'), ('active', 'red'), ('!disabled', 'red')], background=[('pressed', 'yellow'), ('active', 'black'), ('!disabled', 'yellow')])
        
        # Create welcome message and provide basic information 
        ttk.Label(self, text='           Welcome to the Hydrograph Simulator!            ', foreground='red', font='Times 25 italic bold', padding=20).place(x=425, y=120, anchor=tk.CENTER)
        ttk.Label(self, text='Find following...', foreground='red', font='Times 17 italic bold').place(x=40, y=240, anchor=tk.W)
        ttk.Label(self, text='1. Response of a catchment area to a rainfall.', foreground='red', font='Times 17 italic bold').place(x=40, y=275, anchor=tk.W)
        ttk.Label(self, text='2. Peak discharge value for series of rainfall.', foreground='red', font='Times 17 italic bold').place(x=40, y=310, anchor=tk.W)
        ttk.Label(self, text='3. Helps in find out the time difference between peak rainfall and peak discharge.', foreground='red', font='Times 17 italic bold').place(x=40, y=345, anchor=tk.W)
        
        # Navigation buttons at bottom for Exit and next page
        ttk.Button(self, text='Quit!', style='landPageButton.TButton', command=self.exitPage).place(x=40,y=450, anchor=tk.W)
        self.continueButton = ttk.Button(self, text='Continue...', style='landPageButton.TButton', command=self.nextPage).place(x=810,y=450,anchor=tk.E)

    def exitPage(self):
        self.destroy()  # Destroy and exit from the window

    def nextPage(self):
        self.destroy()  # Destory current root window
        nextWindow = InitialInformationCollectionWindow()   # Initialise next root window
        nextWindow.mainloop()



if __name__ == "__main__":
    myApp = App() # Create the instance of the App class
    myApp.mainloop() # Display the root window