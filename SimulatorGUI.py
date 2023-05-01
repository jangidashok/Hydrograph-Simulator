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
    # All required functions for the zoom-in & zoom-out are declared here

    def redefineScale(self):
        # Ensure the proper scale factor before proceeding to next operation
        self.scaleFactorX = (float(self.imageFinalX-self.imageInitialX))/(self.resizedDimensions[0])
        self.scaleFactorY = (float(self.imageFinalY-self.imageInitialY))/(self.resizedDimensions[1])

    def zoomInNextCoordinates(self,x,y):
        # Zoom-in to the 10% of original dimensions
        # Linear interpolation of 10% length from mouse pointer to all four directions
        self.imageInitialX = self.imageInitialX + int((float(x))*(1-1/1.1)*self.scaleFactorX)
        self.imageFinalX = self.imageFinalX - int((float(self.resizedDimensions[0]-x))*(1-1/1.1)*self.scaleFactorX)
        self.imageInitialY = self.imageInitialY + int((float(y))*(1-1/1.1)*self.scaleFactorY)
        self.imageFinalY = self.imageFinalY - int((float(self.resizedDimensions[1]-y))*(1-1/1.1)*self.scaleFactorY)

    def zoomOutNextCoordinates(self,x,y):
        # Zoom-out to the 10% of original dimensions
        # 10% dimension increment from the mouse pointer in all four directions
        self.imageInitialX = self.imageInitialX - int((float(x))*0.1*self.scaleFactorX)
        self.imageFinalX = self.imageFinalX + int((float(self.resizedDimensions[0]-x))*0.1*self.scaleFactorX)
        self.imageInitialY = self.imageInitialY - int((float(y))*0.1*self.scaleFactorY)
        self.imageFinalY = self.imageFinalY + int((float(self.resizedDimensions[1]-y))*0.1*self.scaleFactorY)

        # Handle the corner case where increment is behond the possible values
        self.imageInitialX = max(self.imageInitialX, 0)
        self.imageFinalX = min(self.imageFinalX, self.originalImageWidth)
        self.imageInitialY = max(self.imageInitialY, 0)
        self.imageFinalY = min(self.imageFinalY, self.originalImageHeight)

    def setImageWindowDimensions(self):
        # Initialise the neccessary variables and defining a window of fixed sized for image output
        
        # Below two are top-left corner coordinates which will help in cropping from original image and initialising them
        self.imageInitialX = 0
        self.imageInitialY = 0
        # Below two are bottom-right corner coordinates which will help in cropping from original image and initialising them
        self.imageFinalX = self.newImage.shape[1]
        self.imageFinalY = self.newImage.shape[0]

        # Variables to have original image size
        self.originalImageWidth = self.newImage.shape[1]
        self.originalImageHeight = self.newImage.shape[0]

        # Assuming that window size is 950x650 (i.e. either width is 950 or height is 650)
        # Dimension which is exceeding largest from our desired window would be reduced to our fixed size window and remaining would be adjusted on that same scale
        if(self.originalImageHeight-650 < self.originalImageWidth-950):
            self.scaleFactorX = (float(self.originalImageWidth))/950
            self.scaleFactorY = (float(self.originalImageWidth))/950
        else:
            self.scaleFactorX = (float(self.originalImageHeight))/650
            self.scaleFactorY = (float(self.originalImageHeight))/650

        # As per selected above scales we are fixing a final window to be shown
        self.resizedDimensions = (int(self.originalImageWidth/self.scaleFactorX),int(self.originalImageHeight/self.scaleFactorY))

    def zoomInZoomOut(self, x, y, flags):
        # All required mouse functionality will be dealt here
        if (flags>0):  # This is for zoom-in
            # For accuracy purpose, scale is defined again
            self.redefineScale()
            # Make sure that it is zoomable upto only a limited scale
            if(self.scaleFactorX<0.3 or self.scaleFactorY<0.3):
                return
            # Set new coordinates
            self.zoomInNextCoordinates(x,y)

        else:  # This is for zoom-out
            # For accuracy purpose, scale is defined again
            self.redefineScale()
            # Make sure that it is zoom-out upto a limited value
            if(self.scaleFactorX>10 or self.scaleFactorY>10):
                return
            # Set new coordinates
            self.zoomOutNextCoordinates(x,y)

    def reconstructWindow(self):
        # As per re-coordinating, show updated window for live effects
        self.newImage = cv2.imread(self.imageLocation, 1)
        self.newImage = self.newImage[self.imageInitialY:self.imageFinalY, self.imageInitialX:self.imageFinalX]
        self.newImage = cv2.resize(self.newImage,self.resizedDimensions)



class ProcessSurfaceFlowAndFinalGraphOutput(tk.Tk):

    def __init__(self, fileLocation, outletNode):
        
        # Initialise the window properties
        super().__init__()
        self.title('Outflow Hydrograph...')
        self.geometry('850x400')
        self.minsize(850,420)
        self.maxsize(850,420)
        self.fileLocation = fileLocation
        self.outletNode = outletNode

        currentWorkingDirectory = os.getcwd()
        imageFileLocation = currentWorkingDirectory + '\i1.png'
        pathToIcon = currentWorkingDirectory + r'\guiIcon.ico'
        self.wm_iconbitmap(pathToIcon)
        self.backgroundImageObject = tk.PhotoImage(file = imageFileLocation)

        self.newCanvas = tk.Canvas(self)
        self.newCanvas.create_image(0,0,anchor=tk.NW,image=self.backgroundImageObject)
        self.newCanvas.place(x=0,y=0,relheight=1,relwidth=1)

        self.startCalculating()

        ttk.Label(self, text='Maximum flow rate obtained:', foreground='red', font='Times 17 italic bold').place(x=520,y=100,anchor=tk.W)
        # ttk.Label(self, text='245 m^3/sec', foreground='red', font='Times 17 italic bold').place(x=520,y=150,anchor=tk.W)
        ttk.Label(self, text='And it occured at time, t:', foreground='red', font='Times 17 italic bold').place(x=520,y=250,anchor=tk.W)
        # ttk.Label(self, text='1300 seconds', foreground='red', font='Times 17 italic bold').place(x=520,y=300,anchor=tk.W)

    def startCalculating(self):
        # Declare surface flow variables
        manCoeff = 0.01
        deltaTime = 10
        totalSimulationTime = 7000
        rainFallRate = 0.01/3600
        initialWaterDepth = rainFallRate*deltaTime
        trianglesNodes = []
        triangles = []
        trianglesArea = []
        trianglesCG = []
        trianglesAverageGroundElevation = []
        trianglesWaterLevel = []


        # Process nodes and segments to create triangles
        newFileName = self.fileLocation + '/finalNodes.txt'
        pointCloud = np.loadtxt(newFileName,skiprows=0,ndmin=2)
        xyz = pointCloud[:,:]
        xyFormat = []
        
        for ele in xyz:
            xyFormat.append([ele[0], ele[1]])
            trianglesNodes.append([ele[0], ele[1], ele[2]])

        newFileName = self.fileLocation + '/finalSegments.txt'
        pointCloud = np.loadtxt(newFileName,skiprows=0,ndmin=2)
        segs = pointCloud[:,:]
        mandatorySegments = []

        for ele in segs:
            mandatorySegments.append((int(ele[0]), int(ele[1])))

        triangulatedData = tr.triangulate({'vertices':xyFormat, 'segments':mandatorySegments}, 'pA')
        tG = triangulatedData['triangles']

        #  initialise all variables required for the surface flow
        for ele in tG:
            nodeList = [ele[0], ele[1], ele[2]]
            nodeList.sort()
            triangles.append([nodeList[0], nodeList[1], nodeList[2]])
        
        # Create a neighbour triangle track
        # Start three indices have indicator number which represent as follows: 
        # 1 -> There exist a neighbour traingle, 0 -> No flow boundry, -1 -> water transfer to valley segment
        # Next three indices will point the respective triangle or valley segment, -1 means no data available
        neighbourTriangle = [[0, 0, 0, -1, -1, -1] for i in range(len(triangles))]
        trianglesEdgeLengths = [[0, 0, 0] for i in range(len(triangles))]
        neighbourTriangleCGDistance = [[-1, -1, -1] for i in range(len(triangles))]

        for i in range(len(triangles)):
            x1, y1, z1 = trianglesNodes[triangles[i][0]][0], trianglesNodes[triangles[i][0]][1], trianglesNodes[triangles[i][0]][2]
            x2, y2, z2 = trianglesNodes[triangles[i][1]][0], trianglesNodes[triangles[i][1]][1], trianglesNodes[triangles[i][1]][2]
            x3, y3, z3 = trianglesNodes[triangles[i][2]][0], trianglesNodes[triangles[i][2]][1], trianglesNodes[triangles[i][2]][2]

            area = abs(0.5*(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2)))
            trianglesArea.append(area)
            trianglesCG.append([(x1+x2+x3)/3, (y1+y2+y3)/3])
            trianglesAverageGroundElevation.append((z1+z2+z3)/3)
            trianglesWaterLevel.append(trianglesAverageGroundElevation[i]+initialWaterDepth)

            for j in range(3):
                x1, y1 = trianglesNodes[triangles[i][j]][0], trianglesNodes[triangles[i][j]][1]
                x2, y2 = trianglesNodes[triangles[i][(j+1)%3]][0], trianglesNodes[triangles[i][(j+1)%3]][1]

                trianglesEdgeLengths[i][j] = math.sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1))

                for k in range(len(triangles)):
                    if i==k:
                        continue

                    for l in range(3):
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

                        neighbourTriangle[i][j] = 1
                        neighbourTriangle[i][j+3] = k

        newFileName = self.fileLocation + '/noFlowSegments.txt'
        pointCloud = np.loadtxt(newFileName,skiprows=0,ndmin=2)
        seg = pointCloud[:,:]
        noFlowSegments = []

        for ele in seg:
            nodeList = [int(ele[0]), int(ele[1])]
            nodeList.sort()
            noFlowSegments.append([nodeList[0], nodeList[1]])

        for i in range(len(triangles)):
            for j in range(3):
                for k in range(len(noFlowSegments)):
                    if ((j<2) and (triangles[i][j] == noFlowSegments[k][0]) and (triangles[i][(j+1)%3] == noFlowSegments[k][1])):
                        pass
                    elif ((j==2) and (triangles[i][0] == noFlowSegments[k][0]) and (triangles[i][2] == noFlowSegments[k][1])):
                        pass
                    else:
                        continue

                    neighbourTriangle[i][j] = 0
                    neighbourTriangle[i][j+3] = -1

        newFileName = self.fileLocation + '/fullTransferSegments.txt'
        pointCloud = np.loadtxt(newFileName,skiprows=0,ndmin=2)
        seg = pointCloud[:,:]
        valleySegments = []

        for ele in seg:
            nodeList = [int(ele[0]), int(ele[1])]
            nodeList.sort()
            valleySegments.append([nodeList[0], nodeList[1]])

        valleySegmentsLength = [1e-1]*len(valleySegments)

        for i in range(len(triangles)):
            for j in range(3):
                for k in range(len(valleySegments)):
                    if ((j<2) and (triangles[i][j] == valleySegments[k][0]) and (triangles[i][(j+1)%3] == valleySegments[k][1])):
                        pass
                    elif ((j==2) and (triangles[i][0] == valleySegments[k][0]) and (triangles[i][2] == valleySegments[k][1])):
                        pass
                    else:
                        continue

                    neighbourTriangle[i][j] = -1
                    neighbourTriangle[i][j+3] = k
                    valleySegmentsLength[k] = trianglesEdgeLengths[i][j]

        for i in range(len(triangles)):
            for j in range(3):
                if neighbourTriangle[i][j] == 1:
                    cgx1, cgy1 = trianglesCG[i][0], trianglesCG[i][1]
                    cgx2, cgy2 = trianglesCG[neighbourTriangle[i][j+3]][0], trianglesCG[neighbourTriangle[i][j+3]][1]
                    neighbourTriangleCGDistance[i][j] = math.sqrt((cgx2-cgx1)*(cgx2-cgx1) + (cgy2-cgy1)*(cgy2-cgy1))

        totalIntervals = int(totalSimulationTime/deltaTime) + 10
        valleySegmentsCollectionOnIntervalTime = [[0.0]*len(valleySegments) for i in range(totalIntervals)]

        for i in range(deltaTime, totalSimulationTime+1, deltaTime):
            matA = [[0.0]*len(triangles) for i in range(len(triangles))]
            matB = [0.0]*len(triangles)

            if i>3600:
                rainFallRate = 0

            for j in range(len(triangles)):
                matA[j][j] = trianglesArea[j]/deltaTime
                matB[j] = matB[j] + rainFallRate*trianglesArea[j]

                for k in range(3):

                    if neighbourTriangle[j][k] == 1:
                        maxGroundElevation = max(trianglesAverageGroundElevation[j], trianglesAverageGroundElevation[neighbourTriangle[j][k+3]])
                        edgeLength = trianglesEdgeLengths[j][k]
                        cgCenterToCgCenterDistance = neighbourTriangleCGDistance[j][k]
                        waterLevelDifference = trianglesWaterLevel[neighbourTriangle[j][k+3]] - trianglesWaterLevel[j]
                        if waterLevelDifference > 1e-9:
                            heightOfFlow = trianglesWaterLevel[neighbourTriangle[j][k+3]] - maxGroundElevation
                            Qik = edgeLength/(manCoeff*math.sqrt(cgCenterToCgCenterDistance))*(heightOfFlow**1.67)*math.sqrt(waterLevelDifference)
                            dQik_dzi = -edgeLength/(manCoeff*math.sqrt(cgCenterToCgCenterDistance))*((heightOfFlow**1.67)*0.5/math.sqrt(waterLevelDifference))
                            dQik_dzk = edgeLength/(manCoeff*math.sqrt(cgCenterToCgCenterDistance))*(1.67*(heightOfFlow**0.67)*math.sqrt(waterLevelDifference) + (heightOfFlow**1.67)*0.5/math.sqrt(waterLevelDifference))

                            matA[j][j] = matA[j][j] - dQik_dzi
                            matA[neighbourTriangle[j][k+3]][j] = matA[neighbourTriangle[j][k+3]][j] + dQik_dzi
                            matA[j][neighbourTriangle[j][k+3]] = matA[j][neighbourTriangle[j][k+3]] - dQik_dzk
                            matA[neighbourTriangle[j][k+3]][neighbourTriangle[j][k+3]] = matA[neighbourTriangle[j][k+3]][neighbourTriangle[j][k+3]] + dQik_dzk

                            matB[j] = matB[j] + Qik
                            matB[neighbourTriangle[j][k+3]] = matB[neighbourTriangle[j][k+3]] - Qik
                    
                    elif neighbourTriangle[j][k] == -1:
                        heightOfFlow = trianglesWaterLevel[j] - trianglesAverageGroundElevation[j]
                        edgeLength = trianglesEdgeLengths[j][k]
                        Qik = -0.6*1.706*edgeLength*(heightOfFlow**1.5)
                        dQik_dzi = -0.6*1.706*edgeLength*1.5*math.sqrt(heightOfFlow)
                        matA[j][j] = matA[j][j] - dQik_dzi
                        matB[j] = matB[j] + Qik

                        valleySegmentsCollectionOnIntervalTime[int(i/deltaTime)][neighbourTriangle[j][k+3]] = valleySegmentsCollectionOnIntervalTime[int(i/deltaTime)][neighbourTriangle[j][k+3]] - Qik
            
            try:
                mat1 = np.array(matA)
                mat2 = np.array(matB)
                deltaDifference = np.linalg.solve(mat1, mat2)
            except:
                deltaDifference = [0.0]*len(triangles)

            for j in range(len(deltaDifference)):
                trianglesWaterLevel[j] = trianglesWaterLevel[j] + deltaDifference[j]
                if trianglesWaterLevel[j] < trianglesAverageGroundElevation[j] + rainFallRate*deltaTime:
                    trianglesWaterLevel[j] = trianglesAverageGroundElevation[j] + rainFallRate*deltaTime

        valleyIndicatorNumber = [0]*len(valleySegments)
        listOfStarter = []
        backNodeToSegment = {}

        for i in range(len(valleySegments)):
            if valleySegments[i][0] in backNodeToSegment:
                backNodeToSegment[valleySegments[i][0]].append(i)
            else:
                backNodeToSegment[valleySegments[i][0]] = []
                backNodeToSegment[valleySegments[i][0]].append(i)

            if valleySegments[i][1] in backNodeToSegment:
                backNodeToSegment[valleySegments[i][1]].append(i)
            else:
                backNodeToSegment[valleySegments[i][1]] = []
                backNodeToSegment[valleySegments[i][1]].append(i)

        listOfStarter.append([self.outletNode, 1])

        QueueForProcessing = [self.outletNode]
        topologicalListToSolve = []

        for node1 in QueueForProcessing:
            for seg1 in backNodeToSegment[node1]:
                if valleySegments[seg1][0] != node1 and trianglesNodes[valleySegments[seg1][0]][2] >= trianglesNodes[node1][2] and (valleySegments[seg1][0] not in QueueForProcessing):
                    QueueForProcessing.append(valleySegments[seg1][0])
                    topologicalListToSolve.append([valleySegments[seg1][0], seg1, node1])

                elif valleySegments[seg1][1] != node1 and trianglesNodes[valleySegments[seg1][1]][2] >= trianglesNodes[node1][2] and (valleySegments[seg1][1] not in QueueForProcessing):
                    QueueForProcessing.append(valleySegments[seg1][1])
                    topologicalListToSolve.append([valleySegments[seg1][1], seg1, node1])

        topologicalListToSolve.reverse()
        nodeToNormalNumbering = {}
        cnt = 0
        for nodeNumber in backNodeToSegment:
            nodeToNormalNumbering[nodeNumber] = cnt
            cnt = cnt + 1

        ################################# Specific Code to Kinetic wave #################################
        print(valleySegmentsLength)

        b0 = 25
        mn = 0.035
        partitionOnLength = 10
        partitionOnTime = 10
        dt = (float(deltaTime))/partitionOnTime
        beeta = 0.6
        b1 = beeta - 1

        matA1 = [[1e-6]*partitionOnTime for i in range(len(nodeToNormalNumbering))]
        matA2 = [[1e-6]*partitionOnLength for i in range(len(valleySegments))]
        matB = [[0.0]*partitionOnLength for i in range(partitionOnTime)]
        finalXYGraphPoints = [[],[]]

        for i in range(deltaTime, totalSimulationTime, deltaTime):
            n1, s1, n2, lastDischarge = 0, 0, 0, 0

            for tl in topologicalListToSolve:

                for j in range(len(matB)):
                    for k in range(len(matB[0])):
                        matB[j][k] = 0.0

                n1, s1, n2 = tl[0], tl[1], tl[2]

                dx = valleySegmentsLength[s1]/(float(partitionOnLength-1))

                if dx < 1e-7:
                    for j in range(1, partitionOnTime):
                        matA1[nodeToNormalNumbering[n2]][j] = matA1[nodeToNormalNumbering[n2]][j] + matA1[nodeToNormalNumbering[n1]][j]
                        matA1[nodeToNormalNumbering[n1]][j] = 0
                        continue
                
                s0 = (trianglesNodes[n1][2]-trianglesNodes[n2][2])/valleySegmentsLength[s1]
                if s0<0.01:
                    s0 = 0.01

                alpha = (mn*(b0**0.667)/math.sqrt(s0))**0.6
                
                dtdx = dt/dx

                for j in range(1, partitionOnLength):
                    matB[0][j] = matA2[s1][j]

                for j in range(1, partitionOnTime):
                    matB[j][0] = matA1[nodeToNormalNumbering[n1]][j]

                for j in range(1, partitionOnTime):
                    for k in range(1, partitionOnLength):
                        Qavg = 0.5*(matB[j-1][k] + matB[j][k-1])
                        numo = dtdx*matB[j][k-1] + alpha*beeta*matB[j-1][k]*(Qavg**b1) + dt*(valleySegmentsCollectionOnIntervalTime[int(i/deltaTime)][s1]/(deltaTime*partitionOnLength))
                        deno = dtdx + (alpha*beeta*(Qavg**b1))
                        matB[j][k] = numo/deno

                for j in range(1, partitionOnTime):
                    matA1[nodeToNormalNumbering[n1]][j] = 0
                    matA1[nodeToNormalNumbering[n2]][j] = matA1[nodeToNormalNumbering[n2]][j] + matB[j][partitionOnLength-1]

                for j in range(1, partitionOnLength):
                    matA2[s1][j] = matB[partitionOnTime-1][j]
                
            for j in range(1, partitionOnTime):
                if j+1 == partitionOnTime:
                    lastDischarge = matA1[nodeToNormalNumbering[n2]][j]
                matA1[nodeToNormalNumbering[n2]][j] = 0.0

            finalXYGraphPoints[0].append(i)
            finalXYGraphPoints[1].append(lastDischarge)



        #################################################################################################

        for node1 in listOfStarter:
            for seg1 in backNodeToSegment[node1[0]]:
                if valleyIndicatorNumber[seg1] == 0:
                    valleyIndicatorNumber[seg1] = node1[1]
                    if valleySegments[seg1][0] == node1[0]:
                        listOfStarter.append([valleySegments[seg1][1], node1[1]+1])
                    else:
                        listOfStarter.append([valleySegments[seg1][0], node1[1]+1])
        
        totalIntervals = totalIntervals + len(valleySegments)
        finalOutFlowAtOutlet = [0]*totalIntervals
        timeStamp = []
        maxFlowRate, timeAtMaxFlowRate = 0.0, 0

        for i in range(len(valleySegmentsCollectionOnIntervalTime)):
            for j in range(len(valleySegmentsCollectionOnIntervalTime[i])):
                finalOutFlowAtOutlet[i+valleyIndicatorNumber[j]-1] = finalOutFlowAtOutlet[i+valleyIndicatorNumber[j]-1] + valleySegmentsCollectionOnIntervalTime[i][j]

        for i in range(len(finalOutFlowAtOutlet)):
            timeStamp.append(i*deltaTime)
            if i>0 and i+1<len(finalOutFlowAtOutlet):
                finalOutFlowAtOutlet[i] = (finalOutFlowAtOutlet[i-1] + finalOutFlowAtOutlet[i] + finalOutFlowAtOutlet[i+1])/3
            
            if finalOutFlowAtOutlet[i] > maxFlowRate:
                maxFlowRate = finalOutFlowAtOutlet[i]
                timeAtMaxFlowRate = timeStamp[i]

        # for i in range(len(finalXYGraphPoints[1])):
        #     if finalXYGraphPoints[1][i] > maxFlowRate:
        #         maxFlowRate = finalXYGraphPoints[1][i]
        #         timeAtMaxFlowRate = finalXYGraphPoints[0][i]

        data = {'Time Stamp (in sec)': timeStamp, 'Flow Rate (in m^3/s)': finalOutFlowAtOutlet}
        # print(finalXYGraphPoints[1])
        # data = {'Time Stamp (in sec)': finalXYGraphPoints[0], 'Flow Rate (in m^3/s)': finalXYGraphPoints[1]}
        df = pd.DataFrame(data)

        figure = plt.figure(figsize=(5, 4), dpi=100)
        ax = figure.add_subplot(111)
        line = FigureCanvasTkAgg(figure, self)
        line.get_tk_widget().place(x=260, y=210, anchor=tk.CENTER)
        df = df[['Time Stamp (in sec)', 'Flow Rate (in m^3/s)']].groupby('Time Stamp (in sec)').sum()
        df.plot(kind='line', legend=True, ax=ax, color='b', marker='', linestyle='-', fontsize=10)
        ax.set_title('Outflow Hydrograph')

        ttk.Label(self, text= str(maxFlowRate)+' m^3/sec', foreground='red', font='Times 17 italic bold').place(x=520,y=150,anchor=tk.W)
        ttk.Label(self, text= str(timeAtMaxFlowRate)+' seconds', foreground='red', font='Times 17 italic bold').place(x=520,y=300,anchor=tk.W)



class VisualiseCatchmentAreaAndProceedForFinal(tk.Tk, ZoomInZoomOutFunctionality):

    def __init__(self, imageLoc, scaleFactor):
        
        # Initialise the window properties
        super().__init__()
        self.title('Visualise Catchment Area...')
        self.geometry('850x400')
        self.minsize(850,400)
        self.maxsize(850,400)
        currentWorkingDirectory = os.getcwd()
        imageFileLocation = currentWorkingDirectory + '\i1.png'
        pathToIcon = currentWorkingDirectory + r'\guiIcon.ico'
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

        # Design and positioning of Warning labels
        # self.nodeSelectionWarning = ttk.Label(self, text='test label !!!', foreground='red', font='Times 15 italic bold')
        # self.nodeSelectionWarning.place(x=638,y=190,anchor=tk.CENTER)
        # self.nodeAddWarning1 = ttk.Label(self, text='Please wait, processing...', foreground='red', font='Times 15 italic bold')
        # self.nodeAddWarning1.place(x=212,y=270,anchor=tk.CENTER)
        # self.nodeAddWarning2 = ttk.Label(self, text='Please wait, processing...', foreground='red', font='Times 15 italic bold')
        # self.nodeAddWarning2.place(x=638,y=290,anchor=tk.CENTER)

        # Outlet point should be one of the valley Nodes, so Initilise the menulist
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

        # This variable will keep track whether our data is triangulate or not, 0 here means process has not been started
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
        # Only used for zoom-in or zoom-out
        if(event==10):
            self.zoomInZoomOut(x, y, flags)
        else:
            return
        
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
        try:
            newFileName = self.originalImageFileLocation + fileName + 'Nodes.txt'
            pointCloud = np.loadtxt(newFileName,skiprows=0,ndmin=2)
            xyz = pointCloud[:,:]

            for ele in xyz:
                nodeStore.append([ele[0], ele[1], ele[2], 1])

        except:
            pass

        try:
            newFileName = self.originalImageFileLocation + fileName + 'Segments.txt'
            pointCloud = np.loadtxt(newFileName,skiprows=0,ndmin=2)
            tempSegments = pointCloud[:,:]

            for element in tempSegments:
                segStore.append([int(element[0])-1, int(element[1])-1, int(element[2]), int(element[3]), 1])

        except:
            pass


    def createNewNodeList(self, oldList, newList, subList):
        
        for i in range(len(oldList)):
            if oldList[i][3] == 0:
                subList[i] = 1
            else:
                newList.append([oldList[i][0], oldList[i][1], oldList[i][2]])
            
            if i>0:
                subList[i] = subList[i] + subList[i-1]

    
    def refineSegmentList(self, segList, subList, indNum):

        for i in range(len(segList)):
            if segList[i][2] == indNum:
                segList[i][0] = segList[i][0] - subList[segList[i][0]]
            if segList[i][3] == indNum:
                segList[i][1] = segList[i][1] - subList[segList[i][1]]


    def doubleDeactivateRedundantNodes(self, highNodeList, lowNodeList, firstSegment, secondSegment, highIndNum, lowIndNum):
        for i in range(len(highNodeList)):
            for j in range(len(lowNodeList)):
                if (highNodeList[i][0] == lowNodeList[j][0]) and (highNodeList[i][1] == lowNodeList[j][1]):
                    lowNodeList[j][3] = 0

                    for k in range(len(firstSegment)):
                        if j == firstSegment[k][0] and firstSegment[k][2] == lowIndNum:
                            firstSegment[k][0] = i
                            firstSegment[k][2] = highIndNum

                        if j == firstSegment[k][1] and firstSegment[k][3] == lowIndNum:
                            firstSegment[k][1] = i
                            firstSegment[k][3] = highIndNum

                    for k in range(len(secondSegment)):
                        if j == secondSegment[k][0] and secondSegment[k][2] == lowIndNum:
                            secondSegment[k][0] = i
                            secondSegment[k][2] = highIndNum
                        
                        if j == secondSegment[k][1] and secondSegment[k][3] == lowIndNum:
                            secondSegment[k][1] = i
                            secondSegment[k][3] = highIndNum


    def singleDeactivateRedundantNodes(self, highNodeList, lowNodeList, segList, highIndNum, lowIndNum):
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
        # Iterate over high priority segments
        cnt1 = -1
        for seg in highSegments:
            cnt1 = cnt1 + 1
            if seg[4] == 0:
                continue

            x_mult1, y_mult1, c_const1 = 1.0, 1.0, 0.0
            pt1, pt2 = [], []
            pt1z, pt2z = 0.0, 0.0

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

            if pt1[0] == pt2[0]:
                x_mult1 = -1.0
                y_mult1 = 0.0
                c_const1 = pt1[0]
            elif pt1[1] == pt2[1]:
                x_mult1 = 0.0
                c_const1 = pt1[1]
            else:
                x_mult1 = (pt2[1]-pt1[1])/(pt2[0]-pt1[0])
                c_const1 = y_mult1*pt1[1] - x_mult1*pt1[0]

            # Iterate over low priority segments
            cnt2 = -1
            for j in range(len(lowSegments)):
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

                f1 = (y_mult1*pt3[1]-x_mult1*pt3[0]-c_const1)*(y_mult1*pt4[1]-x_mult1*pt4[0]-c_const1)
                f2 = (y_mult2*pt1[1]-x_mult2*pt1[0]-c_const2)*(y_mult2*pt2[1]-x_mult2*pt2[0]-c_const2)

                if f1 > 0 or f2 > 0:
                    continue
                elif (pt1 == pt3) or (pt1 == pt4) or (pt2 == pt3) or (pt2 == pt4):
                    continue
                elif  (x_mult1 == x_mult2):
                    continue
                else:
                    mat1 = np.array([[y_mult1, -x_mult1], [y_mult2, -x_mult2]])
                    mat2 = np.array([c_const1, c_const2])
                    sol = np.linalg.solve(mat1, mat2)
                    x_sol, y_sol = sol[1], sol[0]

                    if x_sol == pt1[0] and y_sol == pt1[1]:
                        lowSegments[cnt2][4] = 0
                        lowSegments.append([locSeg[0], seg[0], locSeg[2], seg[2], 1])
                        lowSegments.append([seg[0], locSeg[1], seg[2], locSeg[3], 1])
                    elif x_sol == pt2[0] and y_sol == pt2[1]:
                        lowSegments[cnt2][4] = 0
                        lowSegments.append([locSeg[0], seg[1], locSeg[2], seg[3], 1])
                        lowSegments.append([seg[1], locSeg[1], seg[3], locSeg[3], 1])
                    elif x_sol == pt3[0] and y_sol == pt3[1]:
                        highSegments[cnt1][4] = 0
                        highSegments.append([seg[0], locSeg[0], seg[2], locSeg[2], 1])
                        highSegments.append([locSeg[0], seg[1], locSeg[2], seg[3], 1])
                        break
                    elif x_sol == pt4[0] and y_sol == pt4[1]:
                        highSegments[cnt1][4] = 0
                        highSegments.append([seg[0], locSeg[1], seg[2], locSeg[3], 1])
                        highSegments.append([locSeg[1], seg[1], locSeg[3], seg[3], 1])
                        break
                    else:
                        highSegments[cnt1][4] = 0
                        lowSegments[cnt2][4] = 0
                        cnt3 = -1

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

                        lowSegments.append([locSeg[0], cnt3, locSeg[2], highIndNum, 1])
                        lowSegments.append([cnt3, locSeg[1], highIndNum, locSeg[3], 1])
                        highSegments.append([seg[0], cnt3, seg[2], highIndNum, 1])
                        highSegments.append([cnt3, seg[1], highIndNum, seg[3], 1])
                        break


    def absorbHangingSegments(self, oldList, newList):
        for element in oldList:
            if element[4] == 0:
                continue
            newList.append([element[0], element[1], element[2], element[3]])


    def prepareFinalSegmentList(self, catchmentListLength, valleyListLength, ridgeListLength, oldSegmentList, newSegmentList):
        for element in oldSegmentList:
            nextList = []

            if element[0] == element[1] and element[2] == element[3]:
                continue

            if element[2] == 0:
                nextList.append(element[0])
            elif element[2] == 1:
                nextList.append(catchmentListLength + element[0])
            elif element[2] == 2:
                nextList.append(catchmentListLength + valleyListLength + element[0])
            else:
                nextList.append(catchmentListLength + valleyListLength + ridgeListLength + element[0])

            if element[3] == 0:
                nextList.append(element[1])
            elif element[3] == 1:
                nextList.append(catchmentListLength + element[1])
            elif element[3] == 2:
                nextList.append(catchmentListLength + valleyListLength + element[1])
            else:
                nextList.append(catchmentListLength + valleyListLength + ridgeListLength + element[1])

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

        self.initialiseNodesAndSegments('/catchment', tempCatchmentNodes, catchmentSegments)
        self.initialiseNodesAndSegments('/valley', tempValleyNodes, valleySegments)
        self.initialiseNodesAndSegments('/ridge', tempRidgeNodes, ridgeSegments)
        self.initialiseNodesAndSegments('/contour', tempContourNodes, contourSegments)

        self.doubleDeactivateRedundantNodes(tempCatchmentNodes, tempValleyNodes, valleySegments, ridgeSegments, 0, 1)
        self.singleDeactivateRedundantNodes(tempCatchmentNodes, tempRidgeNodes, ridgeSegments, 0, 2)
        self.singleDeactivateRedundantNodes(tempCatchmentNodes, tempContourNodes, contourSegments, 0, 3)

        self.doubleDeactivateRedundantNodes(tempContourNodes, tempValleyNodes, valleySegments, ridgeSegments, 3, 1)
        self.singleDeactivateRedundantNodes(tempContourNodes, tempRidgeNodes, ridgeSegments, 3, 2)

        catchmentNodes = []
        valleyNodes = []
        ridgeNodes = []
        contourNodes = []

        for node in tempCatchmentNodes:
            catchmentNodes.append([node[0], node[1], node[2]])

        downIntervalList = [0]*len(tempValleyNodes)
        self.createNewNodeList(tempValleyNodes, valleyNodes, downIntervalList)
        self.refineSegmentList(valleySegments, downIntervalList, 1)
        self.refineSegmentList(ridgeSegments, downIntervalList, 1)

        downIntervalList = [0]*len(tempRidgeNodes)
        self.createNewNodeList(tempRidgeNodes, ridgeNodes, downIntervalList)
        self.refineSegmentList(ridgeSegments, downIntervalList, 2)

        downIntervalList = [0]*len(tempContourNodes)
        self.createNewNodeList(tempContourNodes, contourNodes, downIntervalList)
        self.refineSegmentList(contourSegments, downIntervalList, 3)
        self.refineSegmentList(valleySegments, downIntervalList, 3)
        self.refineSegmentList(ridgeSegments, downIntervalList, 3)

        self.findPointOfIntersections(catchmentNodes, valleyNodes, ridgeNodes, contourNodes, catchmentSegments, valleySegments, 0)
        self.findPointOfIntersections(catchmentNodes, valleyNodes, ridgeNodes, contourNodes, catchmentSegments, ridgeSegments, 0)
        self.findPointOfIntersections(catchmentNodes, valleyNodes, ridgeNodes, contourNodes, catchmentSegments, contourSegments, 0)

        self.findPointOfIntersections(catchmentNodes, valleyNodes, ridgeNodes, contourNodes, contourSegments, valleySegments, 3)
        self.findPointOfIntersections(catchmentNodes, valleyNodes, ridgeNodes, contourNodes, contourSegments, ridgeSegments, 3)


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

        finalNodes = []
        finalSegments = []
        finalNoFlowSegments = []
        finalFullTransferSegments = []
        
        finalNodes.extend(catchmentNodes), finalNodes.extend(valleyNodes), finalNodes.extend(ridgeNodes), finalNodes.extend(contourNodes)

        self.prepareFinalSegmentList(len(catchmentNodes), len(valleyNodes), len(ridgeNodes), catchmentSegments, finalNoFlowSegments)
        self.prepareFinalSegmentList(len(catchmentNodes), len(valleyNodes), len(ridgeNodes), valleySegments, finalFullTransferSegments)
        self.prepareFinalSegmentList(len(catchmentNodes), len(valleyNodes), len(ridgeNodes), ridgeSegments, finalNoFlowSegments)
        self.prepareFinalSegmentList(len(catchmentNodes), len(valleyNodes), len(ridgeNodes), contourSegments, finalSegments)

        finalSegments.extend(finalNoFlowSegments), finalSegments.extend(finalFullTransferSegments)
        self.totalCatchmentNodes = len(catchmentNodes)

        fileLocation = self.originalImageFileLocation + '/finalNodes.txt'
        with open(fileLocation, 'w') as file:
            for elements in finalNodes:
                file.write(str((float(elements[0]))*self.scaleFactor) + ' ' + str((float(elements[1]))*self.scaleFactor) + ' ' + str(float(elements[2])) + '\n')
            file.close()

        fileLocation = self.originalImageFileLocation + '/noFlowSegments.txt'
        with open(fileLocation, 'w') as file:
            for elements in finalNoFlowSegments:
                tempList = [elements[0], elements[1]]
                tempList.sort()
                file.write(str(tempList[0]) + ' ' + str(tempList[1]) + '\n')
            file.close()

        fileLocation = self.originalImageFileLocation + '/fullTransferSegments.txt'
        with open(fileLocation, 'w') as file:
            for elements in finalFullTransferSegments:
                tempList = [elements[0], elements[1]]
                tempList.sort()
                file.write(str(tempList[0]) + ' ' + str(tempList[1]) + '\n')
            file.close()

        fileLocation = self.originalImageFileLocation + '/finalSegments.txt'
        with open(fileLocation, 'w') as file:
            for elements in finalSegments:
                tempList = [elements[0], elements[1]]
                tempList.sort()
                file.write(str(tempList[0]) + ' ' + str(tempList[1]) + '\n')
            file.close()

        self.isProcessing = 2


    def openGraphInBrowser(self):
        if self.isProcessing == 0:
            self.nodeAddWarning1 = ttk.Label(self, text='Please wait, processing...', foreground='red', font='Times 15 italic bold')
            self.nodeAddWarning1.place(x=212,y=270,anchor=tk.CENTER)
            self.processInputNodesAndSegments()
            self.nodeAddWarning1.destroy()

        elif self.isProcessing == 1:
            self.nodeAddWarning1 = ttk.Label(self, text='Please wait, still processing...', foreground='red', font='Times 15 italic bold')
            self.nodeAddWarning1.place(x=212,y=270,anchor=tk.CENTER)
            while(self.isProcessing == 1):
                pass
            self.nodeAddWarning1.destroy()

        newFileName = self.originalImageFileLocation + '/finalNodes.txt'
        pointCloud = np.loadtxt(newFileName,skiprows=0,ndmin=2)
        xyz = pointCloud[:,:]
        x, y, z = pointCloud[:,0], pointCloud[:,1], pointCloud[:,2]
        xyFormat = []
        nodeSegments = []
        
        for ele in xyz:
            xyFormat.append([ele[0], ele[1]])

        newFileName = self.originalImageFileLocation + '/finalSegments.txt'
        pointCloud = np.loadtxt(newFileName,skiprows=0,ndmin=2)
        segs = pointCloud[:,:]

        for ele in segs:
            nodeSegments.append((int(ele[0]), int(ele[1])))

        triangulatedData = tr.triangulate({'vertices':xyFormat, 'segments':nodeSegments}, 'pA')

        i_, j_, k_ = triangulatedData['triangles'][:,0], triangulatedData['triangles'][:,1], triangulatedData['triangles'][:,2]

        fig = go.Figure(data=[go.Mesh3d(x=x, y=y, z=z, i=i_, j=j_, k=k_, color='cyan', opacity=0.80)])
        f2 = go.FigureWidget(fig)
        f2.show()


    def nextOperations(self):

        if (self.canProceed == 2):
            labelToShow = 'There is No valley Points.'
            self.selectNodeErrorHandling(labelToShow)
            return
        
        elif (self.canProceed == 0):
            labelToShow = 'Please select outlet node!'
            self.selectNodeErrorHandling(labelToShow)
            return
        
        elif self.isProcessing == 0:
            self.nodeAddWarning2 = ttk.Label(self, text='Please wait, processing...', foreground='red', font='Times 15 italic bold')
            self.nodeAddWarning2.place(x=638,y=290,anchor=tk.CENTER)
            self.processInputNodesAndSegments()
            self.nodeAddWarning2.destroy()

        elif self.isProcessing == 1:
            self.nodeAddWarning2 = ttk.Label(self, text='Please wait, still processing...', foreground='red', font='Times 15 italic bold')
            self.nodeAddWarning2.place(x=638,y=290,anchor=tk.CENTER)
            while(self.isProcessing == 1):
                pass
            self.nodeAddWarning2.destroy()

        strToNum = self.Node.get()
        finalOutletNode = self.totalCatchmentNodes + int(strToNum[1:]) - 1
        fileLocations = self.originalImageFileLocation
        self.destroy()

        nextPage = ProcessSurfaceFlowAndFinalGraphOutput(fileLocations, finalOutletNode)
        nextPage.mainloop()



class elevationValueInputFromWindow(tk.Toplevel):
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

        # If window is closed by cross icon
        self.protocol("WM_DELETE_WINDOW", self.testAndProceed)
        self.elevationFromWindow = parent.elevationFromWindow

        ttk.Label(self, text='Elevation: ', foreground='black', font='Times 15 italic bold').place(x=120, y=30, anchor=tk.E)
        ttk.Entry(self, textvariable=self.elevationFromWindow, foreground='black', font='helvetica 13', width=10).place(x=130,y=30,anchor=tk.W)
        # self.warningLabel = ttk.Label(self, text='test label !!!', foreground='red', font='Times 13 italic bold')
        # self.warningLabel.place(x=125,y=60,anchor=tk.CENTER)
        ttk.Button(self, text='Enter', command=self.testAndProceed).place(x=125,y=90,anchor=tk.CENTER)
        
    def testAndProceed(self):
        try:
            self.warningLabel.destroy()
        except:
            pass

        strVar = self.elevationFromWindow.get()

        if(strVar == ''):
            self.warningLabel = ttk.Label(self, text="Enter a numerical value.", foreground='red', font='Times 13 italic bold')
            self.warningLabel.place(x=125,y=60,anchor=tk.CENTER)
            return

        try:
            isFloatable = float(self.elevationFromWindow.get())
            
            if (isFloatable <= 0):
                self.warningLabel = ttk.Label(self, text="Enter a value > 0.", foreground='red', font='Times 13 italic bold')
                self.warningLabel.place(x=125,y=60,anchor=tk.CENTER)
                return
        except:
            self.warningLabel = ttk.Label(self, text="Elevation can't be Non-numerical!", foreground='red', font='Times 13 italic bold')
            self.warningLabel.place(x=125,y=60,anchor=tk.CENTER)
            return
        
        self.quit()
        self.destroy()



class CommonSubWindowProperties(tk.Toplevel):
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

        # Create a copy of image so that later could be replaced
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
    def __init__(self, parent):
        # Set top level window on the previous defined window
        super().__init__(parent)

        # Fix new geometry for this top-level window
        self.geometry('500x540')
        self.title('Valley Lines')

        # Line Separator
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

        # dictionary for node labels
        self.nodeDict = {}
        self.segmentDict1 = {}
        self.segmentDict2 = {}
        self.nodePoint = []

        # Initialise the dictionaries
        try:

            newFileName = self.originalImageFileLocation + '/valleyNodes.txt'
            pointCloud = np.loadtxt(newFileName,skiprows=0,ndmin=2)
            nodes, nodeCoordinates = pointCloud[:,0], pointCloud[:,0:]
            self.currentNumber = len(nodes)+1

            for index in range(0, len(nodes)):
                newNumbering = 'N' + str(int(index+1))
                self.nodeDict[newNumbering] = nodeCoordinates[index]
                self.menulist1.add_radiobutton(label=newNumbering, value=newNumbering, variable=self.Node1, command=self.updateNode1)
                self.menulist2.add_radiobutton(label=newNumbering, value=newNumbering, variable=self.Node2, command=self.updateNode2)

            newFileName = self.originalImageFileLocation + '/valleySegments.txt'
            pointCloud = np.loadtxt(newFileName,skiprows=0,ndmin=2)
            segments, nodesCombination = pointCloud[:,0], pointCloud[:,0:]
            self.currentNumber1 = len(segments)+1

            for index in range(0, len(segments)):
                n1, n2 = 'N' + str(int(nodesCombination[index][0])), 'N' + str(int(nodesCombination[index][1]))
                self.segmentDict2['S' + str(int(index+1))] = [n1, n2]
                self.segmentDict1[n1+n2] = 'S' + str(int(index+1))
        
        except:
            self.currentNumber = 1
            self.currentNumber1 = 1


    def mouseHandlingForNodePoints(self, event, x, y, flags, param):
        if(event==10): # This is for zoom-in or zoom-out
            self.zoomInZoomOut(x, y, flags)
        
        elif event == cv2.EVENT_LBUTTONDOWN or event==cv2.EVENT_RBUTTONDOWN:
            self.redefineScale()
            # Store node
            self.nodePoint.append(self.imageInitialX+(int(x*self.scaleFactorX)))
            self.nodePoint.append(self.imageInitialY+(int(y*self.scaleFactorY)))
            # Window should be closed after selecting the node
            cv2.destroyWindow('Please select node point...')
            return
        
        else:
            return
        
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


    def updateNode1(self):
        self.menuButton1['text'] = self.Node1.get()

    
    def updateNode2(self):
        self.menuButton2['text'] = self.Node2.get()


    def addNewNode(self):
        
        try:
            # If value is conversionable to float then we are good to go
            self.elevation = float(self.elevationValue.get())
            
            if self.elevation <= 0: # Value can't be negative or zero in any case
                labelToShow = 'Enter a value (>0)'
                self.addNodeErrorHandling(labelToShow)
            
            elif len(self.nodePoint) == 0:
                labelToShow = 'Select a node point'
                self.addNodeErrorHandling(labelToShow)

            else:  # Remove flag because we are good to go
                labelToShow = 'None'
                self.addNodeErrorHandling(labelToShow)
                self.newImage = cv2.imread(self.imageLocation, 1)
                self.circleVariable = cv2.circle(self.newImage, (self.nodePoint[0], self.nodePoint[1]), 2, (0, 0, 255), -1)
                cv2.imwrite(self.imageLocation, self.circleVariable)
                newNumbering = 'N' + str(self.currentNumber)
                self.currentNumber = self.currentNumber + 1
                self.textVariable = cv2.putText(self.newImage, newNumbering, (self.nodePoint[0], self.nodePoint[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
                cv2.imwrite(self.imageLocation, self.textVariable)
                self.nodePoint.append(self.elevation)
                self.menulist1.add_radiobutton(label=newNumbering, value=newNumbering, variable=self.Node1, command=self.updateNode1)
                self.menulist2.add_radiobutton(label=newNumbering, value=newNumbering, variable=self.Node2, command=self.updateNode2)
                self.nodeDict[newNumbering] = copy.deepcopy(self.nodePoint)
                self.nodePoint.clear()
                self.elevationValue.set('')

        except:
            # If value entered is not float show corresponding error and put flag to reset it
            labelToShow = 'Enter a float value.'
            self.addNodeErrorHandling(labelToShow)


    def mouseHandlingForViewNodes(self, event, x, y, flags, param):
        if(event==10): # This is for zoom-in or zoom-out
            self.zoomInZoomOut(x, y, flags)

        else:
            return
        
        self.reconstructWindow()
        cv2.imshow('Entered Nodes...', self.newImage)


    def viewNodes(self):
        self.newImage = cv2.imread(self.imageLocation, 1)
        self.setImageWindowDimensions()
        self.newImage = cv2.resize(self.newImage,self.resizedDimensions)
        cv2.imshow('Entered Nodes...', self.newImage)
        cv2.setMouseCallback('Entered Nodes...', self.mouseHandlingForViewNodes)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def selectNode1ErrorHandling(self, newLabel):
        try:
            self.nodeSelectionWarning1.destroy()
        except:
            pass

        if newLabel != 'None':
            self.nodeSelectionWarning1 = ttk.Label(self, text=newLabel, foreground='red', font='helvetica 13')
            self.nodeSelectionWarning1.place(x=125,y=360,anchor=tk.W)


    def selectNode2ErrorHandling(self, newLabel):
        try:
            self.nodeSelectionWarning2.destroy()
        except:
            pass

        if newLabel != 'None':
            self.nodeSelectionWarning2 = ttk.Label(self, text=newLabel, foreground='red', font='helvetica 13')
            self.nodeSelectionWarning2.place(x=375,y=360,anchor=tk.W)


    def selectNode3ErrorHandling(self, newLabel):
        try:
            self.nodeSelectionWarning3.destroy()
        except:
            pass

        if newLabel != 'None':
            self.nodeSelectionWarning3 = ttk.Label(self, text=newLabel, foreground='red', font='helvetica 13')
            self.nodeSelectionWarning3.place(x=250,y=360,anchor=tk.CENTER)


    def addNewEdge(self):
        str1 = self.Node1.get()
        str2 = self.Node2.get()

        labelToShow = 'None'
        self.selectNode1ErrorHandling(labelToShow)
        self.selectNode2ErrorHandling(labelToShow)
        self.selectNode3ErrorHandling(labelToShow)

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

        segmentEnds = []

        if str1 < str2:
            segmentEnds.append(str1)
            segmentEnds.append(str2)
        else:
            segmentEnds.append(str2)
            segmentEnds.append(str1)

        try:
            ifexistPrevious = self.segmentDict1[segmentEnds[0]+segmentEnds[1]]
            labelToShow = f'This segment already exist: {ifexistPrevious}'
            self.selectNode3ErrorHandling(labelToShow)
            return
        except:
            pass
        
        numbering = 'S' + str(self.currentNumber1)
        self.currentNumber1 = self.currentNumber1 + 1

        self.segmentDict1[segmentEnds[0] + segmentEnds[1]] =  numbering
        self.segmentDict2[numbering] = segmentEnds

        self.newImage = cv2.imread(self.imageLocation, 1)
        self.lineVariable = cv2.line(self.newImage, (int(self.nodeDict[str1][0]), int(self.nodeDict[str1][1])), (int(self.nodeDict[str2][0]), int(self.nodeDict[str2][1])), (0,0,255), 1)
        cv2.imwrite(self.imageLocation, self.lineVariable)
        self.textVariable = cv2.putText(self.newImage, numbering, (int((self.nodeDict[str1][0]+self.nodeDict[str2][0])/2), int((self.nodeDict[str1][1]+self.nodeDict[str2][1])/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
        cv2.imwrite(self.imageLocation, self.textVariable)


    def saveAndExit(self):
        os.remove(self.destinationFolder + '/pointsCollection.jpg')
        sop.copy(self.imageLocation, self.destinationFolder)

        fileLocation = self.originalImageFileLocation + '/valleyNodes.txt'
        with open(fileLocation, 'w') as file:
            for nodeName, nodePoint in self.nodeDict.items():
                file.write(str(float(nodePoint[0])) + ' ' + str(float(nodePoint[1])) + ' ' + str(float(nodePoint[2])) + '\n')
            file.close()

        fileLocation = self.originalImageFileLocation + '/valleySegments.txt'
        with open(fileLocation, 'w') as file:
            for segmentName, segmentNodes in self.segmentDict2.items():
                file.write(segmentNodes[0][1:] + ' ' + segmentNodes[1][1:] + ' 1 1\n')
            file.close()

        self.dismissAndExit()



class elevationValueORvalleyNode(tk.Toplevel):
    def __init__(self, parent):
        # Set top level window on the previous defined window
        super().__init__(parent)

        # Set icon for this top-level window
        self.currentWorkingDirectory = os.getcwd()
        self.pathToIcon = self.currentWorkingDirectory + r'\guiIcon.ico'
        self.wm_iconbitmap(self.pathToIcon)
        self.minsize(300,320)
        self.maxsize(300,320)
        self.title("Elevation Or Node")

        # If window is closed by cross icon
        self.protocol("WM_DELETE_WINDOW", self.onCrossSteps)
        self.operationType = parent.operationTypeVar
        self.elevationFromWindow = parent.elevationFromWindow

        # Line design
        self.newCanvas = tk.Canvas(self)
        self.newCanvas.create_line(0, 160, 125, 160, dash=(1,1))
        self.newCanvas.create_line(175, 160, 300, 160, dash=(1,1))
        self.newCanvas.place(x=0,y=0,relheight=1,relwidth=1)
        # General design
        ttk.Style().configure('general.TMenubutton', font=('Times italic bold', 13))
        ttk.Style().map('general.TMenubutton', foreground=[('pressed', 'yellow'), ('active', 'red'), ('!disabled', 'black')], background=[('pressed', 'red'), ('active', 'yellow'), ('!disabled', 'white')])
        
        # Upper Element
        ttk.Label(self, text='Either provide elevation', foreground='red', font='Times 17 italic bold').place(x=150, y=30, anchor=tk.CENTER)
        ttk.Label(self, text='Elevation: ', foreground='black', font='Times 15 italic bold').place(x=145, y=65, anchor=tk.E)
        ttk.Entry(self, textvariable=self.elevationFromWindow, foreground='black', font='helvetica 13', width=10).place(x=155,y=65,anchor=tk.W)
        # self.warningLabel = ttk.Label(self, text='Provide a numerical value!', foreground='red', font='Times 13 italic bold')
        # self.warningLabel.place(x=150,y=90,anchor=tk.CENTER)
        ttk.Button(self, text='Enter', command=self.testAndProceed).place(x=150,y=120,anchor=tk.CENTER)

        # Lower Element
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

        for ele in parent.topThreeNodes:
            newNumbering = 'N' + str(ele)
            self.menulist.add_radiobutton(label=newNumbering, value=newNumbering, variable=self.Node, command=self.updateNode)
        

    def removeLabels(self):
        try:
            self.warningLabel.destroy()
        except:
            pass

        try:
            self.warningLabel1.destroy()
        except:
            pass
    

    def updateNode(self):
        self.removeLabels()
        self.menuButton['text'] = self.Node.get()

    
    def selectFromMenu(self):
        self.removeLabels()
        tempStr = self.Node.get()
        
        if tempStr == '':
            self.warningLabel1 = ttk.Label(self, text='Select appropriate node!', foreground='red', font='Times 13 italic bold')
            self.warningLabel1.place(x=150,y=260,anchor=tk.CENTER)
            return
        
        self.operationType.set(int(tempStr[1:]))
        self.quit()
        self.destroy()


    def onCrossSteps(self):
        self.removeLabels()
        self.warningLabel = ttk.Label(self, text="Mandatory! Either proceed through this.", foreground='red', font='Times 13 italic bold')
        self.warningLabel.place(x=150,y=90,anchor=tk.CENTER)
        self.warningLabel1 = ttk.Label(self, text='Or through this!', foreground='red', font='Times 13 italic bold')
        self.warningLabel1.place(x=150,y=260,anchor=tk.CENTER)


    def testAndProceed(self):

        self.removeLabels()
        strVar = self.elevationFromWindow.get()

        if(strVar == ''):
            self.warningLabel = ttk.Label(self, text="Enter a numerical value!", foreground='red', font='Times 13 italic bold')
            self.warningLabel.place(x=150,y=90,anchor=tk.CENTER)
            return

        try:
            isFloatable = float(self.elevationFromWindow.get())
            
            if (isFloatable <= 0):
                self.warningLabel = ttk.Label(self, text="Enter a value > 0", foreground='red', font='Times 13 italic bold')
                self.warningLabel.place(x=150,y=90,anchor=tk.CENTER)
                return
        except:
            self.warningLabel = ttk.Label(self, text="Elevation can't be Non-numerical!", foreground='red', font='Times 13 italic bold')
            self.warningLabel.place(x=150,y=90,anchor=tk.CENTER)
            return
        
        self.operationType.set(0)
        self.quit()
        self.destroy()



class RidgeLinesWindow(CommonSubWindowProperties, ZoomInZoomOutFunctionality):
    def __init__(self, parent):
        # Set top level window on the previous defined window
        super().__init__(parent)

        # Fix new geometry for this top-level window
        self.geometry('500x320')
        self.title('Ridge Lines')

        # Line Separator
        self.newCanvas = tk.Canvas(self)
        self.newCanvas.create_line(0, 220, 500, 220, dash=(5,1))
        self.newCanvas.create_line(250, 220, 250, 320, dash=(1,1))
        self.newCanvas.place(x=0,y=0,relheight=1,relwidth=1)

        # Data in-take design
        ttk.Label(self, text="Step: Add Ridge lines one after another", foreground='black', font='Times 20 italic bold').place(x=250, y=40, anchor=tk.CENTER)
        ttk.Label(self, text="One ridge line is made of continuous nodes", foreground='red', font='Times 17 italic bold').place(x=250, y=90, anchor=tk.CENTER)
        ttk.Label(self, text='(To save, press enter or close input window)', foreground='black', font='Times 17 italic bold').place(x=250, y=120, anchor=tk.CENTER)
        ttk.Button(self, text='Add ridge line', style='generalButton.TButton', command=self.addNewLine).place(x=250,y=170,anchor=tk.CENTER)
        
        # Bottom Section
        ttk.Button(self, text='Dismiss!', style='navButton.TButton', command=self.dismissAndExit).place(x=125,y=270,anchor=tk.CENTER)
        ttk.Button(self, text='Save!', style='navButton.TButton', command=self.saveAndExit).place(x=375,y=270,anchor=tk.CENTER)

        # Temporary Data Collection
        self.nodeCoordinates = []
        self.lastNodeCoordinates = []
        self.ridgeSegments = []
        self.runIndication = 1
        self.valleyNodes = []
        self.topThreeNodes = []
        self.elevationFromWindow = tk.StringVar()
        self.operationTypeVar = tk.IntVar()

        try:

            newFileName = self.originalImageFileLocation + '/valleyNodes.txt'
            pointCloud = np.loadtxt(newFileName,skiprows=0,ndmin=2)
            nodeCoordinates = pointCloud[:,0:]

            for index in range(0, len(nodeCoordinates)):
                self.valleyNodes.append([nodeCoordinates[index][0], nodeCoordinates[index][1], nodeCoordinates[index][2]])

        except:
            pass

        try:
            newFileName = self.originalImageFileLocation + '/ridgeNodes.txt'
            pointCloud = np.loadtxt(newFileName,skiprows=0,ndmin=2)
            nodes = pointCloud[:,0]
            self.initialCount = len(nodes)

        except:
            self.initialCount = 0


    def mouseHandlingForNodePoints(self, event, x, y, flags, param):
        if(event==10): # This is for zoom-in or zoom-out
            self.zoomInZoomOut(x, y, flags)
        
        elif (event == cv2.EVENT_LBUTTONDOWN or event==cv2.EVENT_RBUTTONDOWN) and self.runIndication == 1:
            self.redefineScale()

            x_ = self.imageInitialX+(int(x*self.scaleFactorX))
            y_ = self.imageInitialY+(int(y*self.scaleFactorY))
            
            self.reconstructWindow()
            cv2.imshow('Please select node points for ridge line...', self.newImage)

            diffAndNum = []
            nn = 1

            for ele in self.valleyNodes:
                diffAndNum.append([(float(x_) - ele[0])*(float(x_) - ele[0]) + (float(y_) - ele[1])*(float(y_) - ele[1]), nn])
                nn = nn + 1

            diffAndNum.sort()
            nn = 0
            self.topThreeNodes.clear()

            for index in range(len(diffAndNum)):
                if nn>2:
                    break
                self.topThreeNodes.append(diffAndNum[index][1])
                nn = nn + 1

            self.runIndication = 0
            self.operationTypeVar.set(-1)
            self.elevationFromWindow.set('')
            eVI = elevationValueORvalleyNode(self)
            eVI.grab_set()
            eVI.mainloop()
            self.runIndication = 1
            self.operationType = self.operationTypeVar.get()

            if self.operationType == 0:

                self.newImage = cv2.imread(self.imageLocation, 1)
                self.circleVariable = cv2.circle(self.newImage, (x_, y_), 2, (0, 0, 255), -1)
                cv2.imwrite(self.imageLocation, self.circleVariable)
                self.nodeCoordinates.append([x_, y_, float(self.elevationFromWindow.get())])

                if len(self.lastNodeCoordinates) > 0:

                    self.newImage = cv2.imread(self.imageLocation, 1)
                    self.lineVariable = cv2.line(self.newImage, (int(self.lastNodeCoordinates[0]), int(self.lastNodeCoordinates[1])), (x_, y_), (0,0,255), 1)
                    cv2.imwrite(self.imageLocation, self.lineVariable)

                    self.ridgeSegments.append([self.lastNodeCoordinates[4], self.initialCount + len(self.nodeCoordinates), self.lastNodeCoordinates[3], 2])
                    
                self.lastNodeCoordinates = [x_, y_, float(self.elevationFromWindow.get()), 2, self.initialCount + len(self.nodeCoordinates)]

            else:
                
                if len(self.lastNodeCoordinates) > 0:

                    self.newImage = cv2.imread(self.imageLocation, 1)
                    self.lineVariable = cv2.line(self.newImage, (int(self.lastNodeCoordinates[0]), int(self.lastNodeCoordinates[1])), (int(self.valleyNodes[self.operationType-1][0]), int(self.valleyNodes[self.operationType-1][1])), (0,0,255), 1)
                    cv2.imwrite(self.imageLocation, self.lineVariable)

                    self.ridgeSegments.append([self.lastNodeCoordinates[4], self.operationType, self.lastNodeCoordinates[3], 1])

                self.lastNodeCoordinates = [int(self.valleyNodes[self.operationType-1][0]), int(self.valleyNodes[self.operationType-1][1]), int(self.valleyNodes[self.operationType-1][2]), 1, self.operationType]
            
            self.reconstructWindow()
            cv2.imshow('Please select node points for ridge line...', self.newImage)

            return
        
        else:
            return
        
        self.reconstructWindow()
        cv2.imshow('Please select node points for ridge line...', self.newImage)


    def addNewLine(self):
        self.lastNodeCoordinates.clear()

        self.newImage = cv2.imread(self.imageLocation, 1)
        self.setImageWindowDimensions()
        self.newImage = cv2.resize(self.newImage,self.resizedDimensions)
        cv2.imshow('Please select node points for ridge line...', self.newImage)
        cv2.setMouseCallback('Please select node points for ridge line...', self.mouseHandlingForNodePoints)
        while(True):
            newKey = cv2.waitKey(0)
            if(self.runIndication and (newKey == -1 or newKey == 13)):
                break
        cv2.destroyAllWindows()


    def saveAndExit(self):
        os.remove(self.destinationFolder+'/pointsCollection.jpg')
        sop.copy(self.imageLocation, self.destinationFolder)

        fileLocation = self.originalImageFileLocation + '/ridgeNodes.txt'
        with open(fileLocation, 'a') as file:
            newFileName = self.originalImageFileLocation + '/ridgeNodes.txt'
            pointCloud = np.loadtxt(newFileName,skiprows=0,ndmin=2)
            nodes = pointCloud[:,0]
            lastCounted = len(nodes)

            for items in self.nodeCoordinates:
                lastCounted = lastCounted + 1
                file.write(str(items[0]) + ' ' + str(items[1]) + ' ' + str(items[2]) + '\n')
            
            file.close()

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

        self.dismissAndExit()



class ContourPointsWindow(CommonSubWindowProperties, ZoomInZoomOutFunctionality):
    def __init__(self, parent):
        # Set top level window on the previous defined window
        super().__init__(parent)

        # Fix new geometry for this top-level window
        self.geometry('500x300')
        self.title('Fix elevation contour')

        # Line Separator
        self.newCanvas = tk.Canvas(self)
        self.newCanvas.create_line(0, 220, 500, 220, dash=(5,1))
        # self.newCanvas.create_line(250, 220, 250, 300, dash=(1,1))
        self.newCanvas.place(x=0,y=0,relheight=1,relwidth=1)

        # Design part
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

        self.initialPointsCollection = []  # Points from the map
        self.finalPoints = [] #lastClass.pointsOnContours
        self.segmentsNodes = []
        
        try:
            newFileName = self.originalImageFileLocation + '/contourNodes.txt'
            pointCloud = np.loadtxt(newFileName,skiprows=0,ndmin=2)
            nodes = pointCloud[:,0]
            self.currentNumber = len(nodes)
        except:
            self.currentNumber = 0


    def mouseHandlingForContourPoints(self, event, x, y, flags, param):
        # For zoom-in or zoom-out
        if(event==10):
            self.zoomInZoomOut(x, y, flags)
        # On left or right click by mouse, that point should be collected
        elif event == cv2.EVENT_LBUTTONDOWN or event==cv2.EVENT_RBUTTONDOWN:
            self.redefineScale()
            ll = []
            ll.append(self.imageInitialX+(int(x*self.scaleFactorX)))
            ll.append(self.imageInitialY+(int(y*self.scaleFactorY)))
            self.initialPointsCollection.append(ll)
            self.newImage = cv2.imread(self.imageLocation, 1)
            self.circleVariable = cv2.circle(self.newImage, (self.imageInitialX+(int(x*self.scaleFactorX)), self.imageInitialY+(int(y*self.scaleFactorY))), 2, (0,255,0), 2)
            cv2.imwrite(self.imageLocation, self.circleVariable)
        # If not hitted of above two, escape from below procedure
        else:
            return
        
        self.reconstructWindow()
        cv2.imshow('Select points on contour line...', self.newImage)


    def selectPointsOnContour(self):
        self.newImage = cv2.imread(self.imageLocation, 1)

        self.setImageWindowDimensions()
        self.initialPointsCollection.clear()

        self.newImage = cv2.resize(self.newImage,self.resizedDimensions)
        cv2.imshow('Select points on contour line...', self.newImage)
        cv2.setMouseCallback('Select points on contour line...', self.mouseHandlingForContourPoints)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def contourPointsErrorHandling(self, newLabel):
        try:
            self.nodeAddWarning.destroy()
        except:
            pass

        if newLabel != 'None':
            self.nodeAddWarning = ttk.Label(self, text=newLabel, foreground='red', font='helvetica 13')
            self.nodeAddWarning.place(x=250,y=120,anchor=tk.CENTER)


    def addNewContour(self):
        labelToShow = 'None'
        self.contourPointsErrorHandling(labelToShow)

        try:
            # If value is conversionable to float then we are good to go
            self.elevation = float(self.elevationValue.get())

            if len(self.initialPointsCollection) == 0:
                labelToShow = 'Select points on contour line'
                self.contourPointsErrorHandling(labelToShow)
                return

        except:
            # If value entered is not float show corresponding error and put flag to reset it
            labelToShow = 'Enter a float value for elevation'
            self.contourPointsErrorHandling(labelToShow)
            return
        
        elevation = float(self.elevationValue.get())
        
        for i in range(0, len(self.initialPointsCollection)):
            
            if i>0:
                ll = []
                ll.append(self.currentNumber)
                ll.append(self.currentNumber+1)
                self.segmentsNodes.append(ll)

                self.newImage = cv2.imread(self.imageLocation, 1)
                self.lineVariable = cv2.line(self.newImage, (self.initialPointsCollection[i-1][0], self.initialPointsCollection[i-1][1]), (self.initialPointsCollection[i][0], self.initialPointsCollection[i][1]), (0,0,255), 1)
                cv2.imwrite(self.imageLocation, self.lineVariable)
            
            self.currentNumber = self.currentNumber + 1
            self.initialPointsCollection[i].append(elevation)

            self.finalPoints.append(self.initialPointsCollection[i])

        self.initialPointsCollection.clear()
        self.elevationValue.set('')


    def saveAndExit(self):
        os.remove(self.destinationFolder + '/pointsCollection.jpg')
        sop.copy(self.imageLocation, self.destinationFolder)

        fileLocation = self.originalImageFileLocation + '/contourNodes.txt'
        with open(fileLocation, 'a') as file:
            for items in self.finalPoints:
                file.write(str(items[0]) + ' ' + str(items[1]) + ' ' + str(items[2]) + '\n')
            file.close()

        fileLocation = self.originalImageFileLocation + '/contourSegments.txt'
        with open(fileLocation, 'a') as file:
            for items in self.segmentsNodes:
                file.write(str(items[0]) + ' ' + str(items[1]) + ' 3 3\n')
            file.close()

        self.dismissAndExit()



class PointsCollectionWindow(tk.Tk, ZoomInZoomOutFunctionality):
    def __init__(self, imageAddress, scaleMultiple): #imageAddress, scaleMultiple
        # Set top level window on the previous defined window
        super().__init__()

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
        self.lastCatchmentPoint = []
        self.firstCatchmentPoint = []
        self.cntNum = 0
        self.runIndication = 1
        self.catchmentNodes = []
        self.catchmentSegments = []

    
    def mouseHandlingForCatchmentPoints(self, event, x, y, flags, param):
        if(event==10): # This is for zoom-in or zoom-out
            self.zoomInZoomOut(x, y, flags)
        
        elif (event == cv2.EVENT_LBUTTONDOWN or event==cv2.EVENT_RBUTTONDOWN) and self.runIndication == 1:
            # On left or right click by mouse, that point should be collected
            self.redefineScale()
            self.newImage = cv2.imread(self.imageLocation, 1)

            if len(self.firstCatchmentPoint) == 0:
                self.firstCatchmentPoint.append(self.imageInitialX+(int(x*self.scaleFactorX)))
                self.firstCatchmentPoint.append(self.imageInitialY+(int(y*self.scaleFactorY)))
            
            if len(self.lastCatchmentPoint) > 0:
                self.lineVariable = cv2.line(self.newImage, (self.lastCatchmentPoint[0], self.lastCatchmentPoint[1]), (self.imageInitialX+(int(x*self.scaleFactorX)), self.imageInitialY+(int(y*self.scaleFactorY))), (0,0,255), 1)
                cv2.imwrite(self.imageLocation, self.lineVariable)
                self.lastCatchmentPoint.clear()

            self.lastCatchmentPoint.append(self.imageInitialX+(int(x*self.scaleFactorX)))
            self.lastCatchmentPoint.append(self.imageInitialY+(int(y*self.scaleFactorY)))

            self.circleVariable = cv2.circle(self.newImage, (self.imageInitialX+(int(x*self.scaleFactorX)), self.imageInitialY+(int(y*self.scaleFactorY))), 2, (0,255,0), 2)
            cv2.imwrite(self.imageLocation, self.circleVariable)

            self.reconstructWindow()
            cv2.imshow('Please select boundry points...', self.newImage)

            self.runIndication = 0
            self.elevationFromWindow = tk.StringVar()
            eVI = elevationValueInputFromWindow(self)
            eVI.mainloop()
            self.runIndication = 1

            self.lastCatchmentPoint.append(float(self.elevationFromWindow.get()))
            self.catchmentNodes.append(copy.deepcopy(self.lastCatchmentPoint))

            return
        
        else:
            return
        
        self.reconstructWindow()
        cv2.imshow('Please select boundry points...', self.newImage)


    def drawCatchment(self):
        self.lastCatchmentPoint.clear()
        self.firstCatchmentPoint.clear()

        self.newImage = cv2.imread(self.imageLocation, 1)
        self.setImageWindowDimensions()

        self.newImage = cv2.resize(self.newImage,self.resizedDimensions)
        cv2.imshow('Please select boundry points...', self.newImage)
        cv2.setMouseCallback('Please select boundry points...', self.mouseHandlingForCatchmentPoints)
        while(True):
            newKey = cv2.waitKey(0)
            if(self.runIndication and (newKey == -1 or newKey == 13)):
                break
        cv2.destroyAllWindows()

        if len(self.firstCatchmentPoint)>0:
            self.redefineScale()
            self.newImage = cv2.imread(self.imageLocation, 1)
            self.lineVariable = cv2.line(self.newImage, (self.lastCatchmentPoint[0], self.lastCatchmentPoint[1]), (self.firstCatchmentPoint[0], self.firstCatchmentPoint[1]), (0,0,255), 1)
            cv2.imwrite(self.imageLocation, self.lineVariable)
            
            fileLocation = self.originalImageFileLocation + '/catchmentNodes.txt'
            with open(fileLocation, 'w') as file:
                for items in self.catchmentNodes:
                    file.write(str(items[0]) + ' ' + str(items[1]) + ' ' + str(items[2]) + '\n')
                file.close()

            fileLocation = self.originalImageFileLocation + '/catchmentSegments.txt'
            with open(fileLocation, 'w') as file:
                totSize = len(self.catchmentNodes)
                for i in range(totSize-1):
                    file.write(str(i+1) + ' ' + str(i+2) + ' 0 0\n')
                if totSize>2:
                    file.write(str(1) + ' ' + str(totSize) + ' 0 0\n')
                file.close()


    def resetCondition(self):
        try:
            os.remove(self.originalImageFileLocation + '/pointsCollection.jpg')
            sop.copy(self.originalImageFileLocation + '/nocombosvposrp/pointsCollection.jpg', self.originalImageFileLocation)
            sop.rmtree(self.originalImageFileLocation + '/nocombosvposrp')
        except:
            pass

    def selectValleyPoints(self):
        sVP = ValleyLinesWindow(self)
        sVP.grab_set()
        sVP.mainloop()
        self.resetCondition()


    def selectRidgePoints(self):
        sRP = RidgeLinesWindow(self)
        sRP.grab_set()
        sRP.mainloop()
        self.resetCondition()


    def selectContourPoints(self):
        sCP = ContourPointsWindow(self)
        sCP.grab_set()
        sCP.mainloop()
        self.resetCondition()


    def destroyCurrentWindow(self):
        self.destroy()


    def saveAndClose(self):
        imageLoc = self.originalImageFileLocation
        scaleMultiple = self.scaleMultiple
        self.destroy()
        nextPage = VisualiseCatchmentAreaAndProceedForFinal(imageLoc, scaleMultiple)
        nextPage.mainloop()



class InitialInformationCollectionWindow(tk.Tk, ZoomInZoomOutFunctionality):

    def __init__(self):
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

        # track current working directory and form new image object for background
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
            # Below to show a bullet point on the selected point
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
    # This class is inherited from the tk.Tk class to create root window for our simulator

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
        self.destroy()

    def nextPage(self):
        self.destroy()
        nextWindow = InitialInformationCollectionWindow()
        nextWindow.mainloop()

        # os.chdir(r'C:\Users\DELL\Downloads\BTP-II')
        # sop.rmtree('tempFolder')



if __name__ == "__main__":
    # myApp = App() # Create the instance of the App class
    # myApp.mainloop() # Display the root window
    # newWindow = tk.Tk()
    # newWindow.elevationFromWindow = tk.StringVar()
    # eVIFW = elevationValueInputFromWindow(newWindow)
    # eVIFW.mainloop()
    # print(newWindow.elevationFromWindow.get())
    # newWindow.mainloop()
    # newWindow.title('AF Simulator')
    # newWindow.geometry('400x400')
    # newWindow.requiredFileLocation = r'C:\Users\DELL\Downloads\BTP-II'
    # newWindow.scaleMultiple = 1.5
    # imageLoc = 'C:/Users/DELL/Downloads/BTP-II/trial/FinalOutputFromOutlet'
    imageLoc = 'C:/Users/DELL/Downloads/BTP-II/trial/OneFinalGo'
    # # vLW = VisualiseCatchmentAreaAndProceedForFinal(imageLoc)
    # # vLW.mainloop()

    app = ProcessSurfaceFlowAndFinalGraphOutput(imageLoc, 30)
    app.mainloop()