import numpy as np
from scipy.signal import savgol_filter
from scipy.stats import chisquare

def solve_quadratic(a, b, c):
    if a != 0:
        return (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)
    else:
        return -c / b

def calculateINL(adcBins):
    idealMids = 0.5 + np.arange(len(adcBins[:-1]))
    adcMids = 0.5 * (adcBins[1:] + adcBins[:-1])
    return idealMids - adcMids

def calculateDNL(adcBins):
    return (adcBins[1:] - adcBins[:-1]) - 1

def buildADCBinsFromDNL(dnl):
    rightmostEdge = len(dnl+1)
    bins = np.concatenate(([0], np.cumsum(1+dnl)))
    bins *= rightmostEdge / bins[-1]
    return bins

def buildADCBinsFromINL(inl):
    adcRange = 2**18
    edges = [0]
    
    for i, inlVal in enumerate(inl):
        edges.append(2*(i+1/2-inlVal) - edges[-1])
        
    edges = np.array(edges)
    edges *= (adcRange)/edges[-1]
    return edges

class ADCconstructor():
    def __init__(self, measuredDist, expectedDist, minCounts=100, xCutoff=0):
        useable = expectedDist > minCounts

        self.measuredDist = measuredDist
        self.expectedDist = expectedDist
        self.codes = np.arange(len(measuredDist))
        self.pdf = self.buildPDFfromFilter()
        self._minCounts = minCounts
        self._xCutoff = xCutoff

        firstCode = max(self.codes[useable].min(), xCutoff)
        lastCode = self.codes[useable].max()

        if lastCode < firstCode:
            raise ValueError("firstCode cannot be greater than lastCode. Perhaps xCutoff is too high.")
        
        self.workingCodes = np.arange(firstCode, lastCode+1)

    def buildADCEdges(self, mode="riemann"):
        measuredDist = self.measuredDist
        minCounts = self._minCounts
        xCutoff = self._xCutoff
        preCodes = np.arange(self.workingCodes[0])
        postCodes = np.arange(self.workingCodes[-1]+1, self.codes.max()+1)
       
        if mode == "riemann":
            edges  =  self._buildEdgesRiemann()
        elif mode == "trapazoid":
            edges  =  self._buildEdgesTrapazoid()

        adcBins = np.concatenate((preCodes, edges, postCodes+1))
        return adcBins

    def _buildEdgesRiemann(self):
        codes = self.codes
        workingCodes = self.workingCodes
        measuredDist = self.measuredDist
        expectedDist = self.expectedDist

        firstCode, lastCode = self.workingCodes[0], self.workingCodes[-1]

        if measuredDist[firstCode] > expectedDist[firstCode]:
            codeWidth = 1 + 2 * ((measuredDist[firstCode] - expectedDist[firstCode])/
                                 (expectedDist[firstCode+1] +
                                  expectedDist[firstCode-1]))
        else:
            codeWidth = measuredDist[firstCode]/expectedDist[firstCode]

        edges = np.nan * np.arange(len(workingCodes) + 1)
        edges[0] = firstCode + 0.5 - codeWidth/2
        edges[1] = firstCode + 0.5 + codeWidth/2
        
        for iCode, code in enumerate(workingCodes[1:]):
            leftEdge = edges[iCode+1]
            floorLE = int(np.floor(leftEdge))
            nextInt = floorLE + 1

            forwardArea = self._computeForwardArea(leftEdge, "riemann")
           
            # This is an integer upper bound on where to place the right edge for the code
            # The right edge will lie between the integers corresponding to 
            # nextInt+i-2 and nextInt+i-1
            i = np.argwhere(forwardArea > measuredDist[code]).min()
 
            # This is the integrated area under the filtered curve right of the leftEdge
            # and left of floor(rightEdge). It 0 in the case where floor(rightEdge) < leftEdge
            integratedArea = forwardArea[i-1]
 
            # The integrated area under the filtered curve between max(floor(rightEdge), leftEdge)
            # and rightEdge is: (rightEdge - max(nextInt+i-2, leftEdge)) * 
            # expectedDist[max(nextInt+i-2, floor(leftEdge))]
            # Invoking the max function for the case where leftEdge > floor(rightEdge) (c=0 case)\
 
            # measuredDist[code] = integratedArea + (rightEdge - max(nextInt+i-2, leftEdge) *
            # expectedDist[max(nextInt+i-2, floor(leftEdge))]) gives
 
            rightEdge = ((measuredDist[code] - integratedArea) /
                         (expectedDist[max(nextInt+i-2, floorLE)]) +
                          max(nextInt+i-2, leftEdge))
                
            edges[iCode+2] = rightEdge

        return edges

    def _buildEdgesTrapazoid(self):
        codes = self.codes
        workingCodes = self.workingCodes
        measuredDist = self.measuredDist
        expectedDist = self.expectedDist

        firstCode, lastCode = self.workingCodes[0], self.workingCodes[-1]

        pdf = self.pdf
        slopes = pdf[1:] - pdf[:-1]
     
        if measuredDist[firstCode] > expectedDist[firstCode]:
            a = slopes[firstCode+1] - slopes[firstCode-1]
            b = 2*(pdf[firstCode+1] + pdf[firstCode])
            c = pdf[firstCode+1] + pdf[firstCode] - 2*measuredDist[firstCode]

            dx = solve_quadratic(a, b, c)

            codeWidth = 1 + 2*dx
        else:
            codeWidth = measuredDist[firstCode]/expectedDist[firstCode]
    
        edges = np.nan * np.arange(len(workingCodes) + 1)
        edges[0] = firstCode + 0.5 - codeWidth/2
        edges[1] = firstCode + 0.5 + codeWidth/2

        for iCode, code in enumerate(workingCodes[1:]):
            leftEdge = edges[iCode+1]
            floorLE = int(np.floor(leftEdge))
            nextInt = floorLE + 1

            forwardArea = self._computeForwardArea(leftEdge, "trapazoid")
           
            # This is an integer upper bound on where to place the right edge for the code
            # The right edge will lie between the integers corresponding to 
            # nextInt+i-2 and nextInt+i-1
            i = np.argwhere(forwardArea > measuredDist[code]).min()
            
            # This is the integrated area under the filtered curve right of the leftEdge
            # and left of floor(rightEdge). It 0 in the case where floor(rightEdge) < leftEdge
            integratedArea = forwardArea[i-1]
            a = slopes[nextInt+i-2]
            b = 2 * pdf[nextInt+i-2]
            c = 2 * (integratedArea - measuredDist[code])
            dx = solve_quadratic(a, b, c)
            rightEdge = dx + max(nextInt+i-2, leftEdge)

            edges[iCode+2] = rightEdge

        return edges
 
    def _computeForwardArea(self, leftEdge, mode):
        nextInt = int(np.ceil(leftEdge))
        dx = (nextInt-leftEdge)
        expectedDist = self.expectedDist
        
        # The cumsum of this is the area under the filtered curve to the right
        # of the left edge for this code
        # 0 is prepended for the case where the right edge for this code should be
        # placed before the ceiling of the left edge.
        if mode == "riemann":
            firstRectangleArea = dx*expectedDist[nextInt-1]
            if firstRectangleArea != 0:
                dForwardArea = np.concatenate(([0], [firstRectangleArea],
                                              expectedDist[nextInt:]))
            else:
                dForwardArea = np.concatenate(([firstRectangleArea],
                                              expectedDist[nextInt:]))

        elif mode == "trapazoid":
            slopes = self.pdf[1:] - self.pdf[:-1]
            firstTrapazoidArea = dx * (2 * self.pdf[nextInt] - 
                                       slopes[nextInt-1] * dx) / 2
            if firstTrapazoidArea != 0:
                dForwardArea = np.concatenate(([0], [firstTrapazoidArea],
                                              expectedDist[nextInt:]))
            else:
                dForwardArea = np.concatenate(([firstTrapazoidArea],
                                              expectedDist[nextInt:]))
            
        forwardArea = np.cumsum(dForwardArea)
        return forwardArea

    def buildPDFfromFilter(self):
        filteredDist = self.expectedDist
        pdf = np.zeros(len(filteredDist)+1)

        for i, f in enumerate(filteredDist):
            pdf[i+1] = 2*f - pdf[i]
        
        return pdf

def periodicDNL(bins, amp=0.1):
    n = 18
    bits = np.arange(n)
    bitAmplitudes = 0.1 * np.exp(-bits)
    freqs = np.pi / ((bits + 1))
    dnl = sum(bitAmp*np.cos(freq*bins) for 
              bitAmp, freq in zip(bitAmplitudes, freqs))
    return dnl

def sinusoid(amp=1, freq=1):
    adcRange=2**18
    x = np.arange(adcRange)
    y = amp * np.sin(freq * x)
    return y

def applyADCtoDistributionBasic(adcBins, distribution):
    dnl = calculateDNL(adcBins)
    dnlDistribution = (1 + dnl) * distribution
    return dnlDistribution

def applyADCtoDistributionSophisticated(adcBins, distribution, minCounts=100):
    mask = distribution > minCounts
    binIdxs = np.nonzero(mask)[0]
    x = adcBins[:-1][mask]
    dnlDistribution = distribution * np.nan
    
    for iEdge, binEdge in enumerate(x):
        binIdx = binIdxs[iEdge]
        nextBinEdge = adcBins[binIdx+1]
        lowerLim = int(np.floor(binEdge))
        upperLim = int(np.floor(nextBinEdge))
        
        # Integers between floor of binEdge and nextBinEdge
        interiorInts = np.arange(lowerLim, upperLim+1)
        
        try:
            xx = np.union1d(interiorInts[1:], [binEdge, nextBinEdge])
        except IndexError:
            xx = np.array([binEdge, nextBinEdge])

        y = distribution[interiorInts]
        dx = xx[1:] - xx[:-1]
        rectangleAreas = y * dx
        integral = rectangleAreas.sum()
        dnlDistribution[binIdx] = integral

    return dnlDistribution
