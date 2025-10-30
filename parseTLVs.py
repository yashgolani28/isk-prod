import struct
import sys
import serial
import binascii
import time
import numpy as np
import math
import os
import datetime

# Local File Imports
from gui_common import *

# --- Radar constants (computed from config if available) ---
_C_MPS = 299_792_458.0
try:
    from config_utils import load_config
    _cfg = load_config()
    _fc_ghz = float(_cfg.get("radar", {}).get("center_frequency_ghz", 60.0))
    _SAFE_MAX_VEL_MS  = float(_cfg.get("pretrack_safe_max_vel_ms", 0) or 0)
    _SAFE_MAX_RANGE_M = float(_cfg.get("pretrack_safe_max_range_m", 0) or 0)
except Exception:
    _fc_ghz = 60.0
    _SAFE_MAX_VEL_MS  = 0.0
    _SAFE_MAX_RANGE_M = 0.0
_LAMBDA_M = _C_MPS / (_fc_ghz * 1e9)

# ================================================== Common Helper Functions ==================================================

# Convert 3D Spherical Points to Cartesian
# Assumes sphericalPointCloud is an numpy array with at LEAST 3 dimensions
# Order should be Range, Elevation, Azimuth
def sphericalToCartesianPointCloud(sphericalPointCloud):
    shape = sphericalPointCloud.shape
    cartestianPointCloud = sphericalPointCloud.copy()
    if (shape[1] < 3):
        print('Error: Failed to convert spherical point cloud to cartesian due to numpy array with too few dimensions')
        return sphericalPointCloud

    # Compute X
    # Range * sin (azimuth) * cos (elevation)
    cartestianPointCloud[:,0] = sphericalPointCloud[:,0] * np.sin(sphericalPointCloud[:,1]) * np.cos(sphericalPointCloud[:,2]) 
    # Compute Y
    # Range * cos (azimuth) * cos (elevation)
    cartestianPointCloud[:,1] = sphericalPointCloud[:,0] * np.cos(sphericalPointCloud[:,1]) * np.cos(sphericalPointCloud[:,2]) 
    # Compute Z
    # Range * sin (elevation)
    cartestianPointCloud[:,2] = sphericalPointCloud[:,0] * np.sin(sphericalPointCloud[:,2])
    return cartestianPointCloud


# ================================================== Parsing Function For Individual TLV's ==================================================

# Point Cloud TLV from SDK
def parsePointCloudTLV(tlvData, tlvLength, pointCloud):
    pointStruct = '4f'  # X, Y, Z, and Doppler
    pointStructSize = struct.calcsize(pointStruct)
    numPoints = int(tlvLength/pointStructSize)

    for i in range(numPoints):
        try:
            x, y, z, doppler = struct.unpack(pointStruct, tlvData[:pointStructSize])
        except:
            numPoints = i
            print('Error: Point Cloud TLV Parser Failed')
            break
        tlvData = tlvData[pointStructSize:]
        pointCloud[i,0] = x 
        pointCloud[i,1] = y
        pointCloud[i,2] = z
        pointCloud[i,3] = doppler
    return numPoints, pointCloud

# Side info TLV from SDK
def parseSideInfoTLV(tlvData, tlvLength, pointCloud):
    pointStruct = '2H'  # Two unsigned shorts: SNR and Noise
    pointStructSize = struct.calcsize(pointStruct)
    numPoints = int(tlvLength/pointStructSize)

    for i in range(numPoints):
        try:
            snr, noise = struct.unpack(pointStruct, tlvData[:pointStructSize])
        except:
            numPoints = i
            print('Error: Side Info TLV Parser Failed')
            break
        tlvData = tlvData[pointStructSize:]
        # SNR and Noise are sent as uint16_t which are measured in 0.1 dB Steps
        pointCloud[i,4] = snr * 0.1
        pointCloud[i,5] = noise * 0.1 
    return pointCloud

# Occupancy state machine TLV from small obstacle detection
def parseOccStateMachTLV(tlvData):
    occStateMachOutput = [False] * 32 # Initialize to 32 empty zones
    occStateMachStruct = 'I' # Single uint32_t which holds 32 booleans
    occStateMachLength = struct.calcsize(occStateMachStruct)
    try:
        occStateMachData = struct.unpack(occStateMachStruct, tlvData[:occStateMachLength])
        for i in range(32):
            # Since the occupied/not occupied flags are individual bits in a uint32, mask out each flag one at a time
            occStateMachOutput[i] = ((occStateMachData[0] & (1 << i)) != 0)
    except Exception as e:
        print('Error: Occupancy State Machine TLV Parser Failed')
        print(e)
        return None
    return occStateMachOutput

# Spherical Point Cloud TLV Parser
# MMWDEMO_OUTPUT_MSG_SPHERICAL_POINTS
def parseSphericalPointCloudTLV(tlvData, tlvLength, pointCloud):
    pointStruct = '4f'  # Range, Azimuth, Elevation, and Doppler
    pointStructSize = struct.calcsize(pointStruct)
    numPoints = int(tlvLength/pointStructSize)

    for i in range(numPoints):
        try:
            rng, azimuth, elevation, doppler = struct.unpack(pointStruct, tlvData[:pointStructSize])
        except:
            numPoints = i
            print('Error: Point Cloud TLV Parser Failed')
            break
        tlvData = tlvData[pointStructSize:]
        pointCloud[i,0] = rng
        pointCloud[i,1] = azimuth
        pointCloud[i,2] = elevation
        pointCloud[i,3] = doppler
    
    # Convert from spherical to cartesian
    pointCloud[:,0:3] = sphericalToCartesianPointCloud(pointCloud[:, 0:3])
    return numPoints, pointCloud

# Point Cloud TLV from Capon Chain
# MMWDEMO_OUTPUT_MSG_COMPRESSED_POINTS
def parseCompressedSphericalPointCloudTLV(tlvData, tlvLength, pointCloud):
    pUnitStruct = '5f' # Units for the 5 results to decompress them
    pointStruct = '2bh2H' # Elevation, Azimuth, Doppler, Range, SNR
    pUnitSize = struct.calcsize(pUnitStruct)
    pointSize = struct.calcsize(pointStruct)

    # Parse the decompression factors
    try:
        pUnit = struct.unpack(pUnitStruct, tlvData[:pUnitSize])
    except:
            print('Error: Point Cloud TLV Parser Failed')
            return 0, pointCloud
    # Update data pointer
    tlvData = tlvData[pUnitSize:]

    # Parse each point
    numPoints = int((tlvLength-pUnitSize)/pointSize)
    for i in range(numPoints):
        try:
            elevation, azimuth, doppler, rng, snr = struct.unpack(pointStruct, tlvData[:pointSize])
        except:
            numPoints = i
            print('Error: Point Cloud TLV Parser Failed')
            break
        
        tlvData = tlvData[pointSize:]
        if (azimuth >= 128):
            print ('Az greater than 127')
            azimuth -= 256
        if (elevation >= 128):
            print ('Elev greater than 127')
            elevation -= 256
        if (doppler >= 32768):
            print ('Doppler greater than 32768')
            doppler -= 65536
        # Decompress values
        pointCloud[i,0] = rng * pUnit[3]          # Range
        pointCloud[i,1] = azimuth * pUnit[1]      # Azimuth
        pointCloud[i,2] = elevation * pUnit[0]    # Elevation
        pointCloud[i,3] = doppler * pUnit[2]      # Doppler
        pointCloud[i,4] = snr * pUnit[4]          # SNR

    # Convert from spherical to cartesian
    pointCloud[:,0:3] = sphericalToCartesianPointCloud(pointCloud[:, 0:3])
    return numPoints, pointCloud


# Decode 3D People Counting Target List TLV
# MMWDEMO_OUTPUT_MSG_TRACKERPROC_3D_TARGET_LIST
#3D Struct format
#uint32_t     tid;     /*! @brief   tracking ID */
#float        posX;    /*! @brief   Detected target X coordinate, in m */
#float        posY;    /*! @brief   Detected target Y coordinate, in m */
#float        posZ;    /*! @brief   Detected target Z coordinate, in m */
#float        velX;    /*! @brief   Detected target X velocity, in m/s */
#float        velY;    /*! @brief   Detected target Y velocity, in m/s */
#float        velZ;    /*! @brief   Detected target Z velocity, in m/s */
#float        accX;    /*! @brief   Detected target X acceleration, in m/s2 */
#float        accY;    /*! @brief   Detected target Y acceleration, in m/s2 */
#float        accZ;    /*! @brief   Detected target Z acceleration, in m/s2 */
#float        ec[16];  /*! @brief   Target Error covarience matrix, [4x4 float], in row major order, range, azimuth, elev, doppler */
#float        g;
#float        confidenceLevel;    /*! @brief   Tracker confidence metric*/
def parseTrackTLV(tlvData, tlvLength):
    targetStruct = 'I27f'
    targetSize = struct.calcsize(targetStruct)
    targets = []

    if tlvLength < targetSize:
        print(f"[WARN] TLV too small. len={tlvLength}, expected={targetSize}")
        return targets

    if tlvLength == targetSize:
        try:
            targetData = struct.unpack(targetStruct, tlvData)
            # --- sanity guard (single target fast-path) ---
            px, py, pz = float(targetData[1]), float(targetData[2]), float(targetData[3])
            vx, vy, vz = float(targetData[4]), float(targetData[5]), float(targetData[6])
            vals = (px, py, pz, vx, vy, vz)
            if not all(math.isfinite(v) for v in vals):
                return []
            if _SAFE_MAX_VEL_MS and max(abs(vx), abs(vy), abs(vz)) > _SAFE_MAX_VEL_MS:
                return []
            rng = float(np.sqrt(px*px + py*py + pz*pz))
            if _SAFE_MAX_RANGE_M and rng > _SAFE_MAX_RANGE_M:
                return []

            targets.append({
                "id": int(targetData[0]),
                "posX": float(targetData[1]),
                "posY": float(targetData[2]),
                "posZ": float(targetData[3]),
                "velX": float(targetData[4]),
                "velY": float(targetData[5]),
                "velZ": float(targetData[6]),
                "accX": float(targetData[7]),
                "accY": float(targetData[8]),
                "accZ": float(targetData[9]),
                "g": float(targetData[26]),
                "confidence": float(targetData[27]),
                "speed_kmh": float(np.sqrt(targetData[4]**2 + targetData[5]**2 + targetData[6]**2) * 3.6),
                "distance": float(np.sqrt(targetData[1]**2 + targetData[2]**2 + targetData[3]**2)),
                "azimuth": float(np.degrees(np.arctan2(targetData[1], targetData[2] + 1e-6))),
                "elevation": float(np.degrees(np.arctan2(
                    targetData[3], np.sqrt(targetData[1]**2 + targetData[2]**2 + 1e-6)))),
                "snr": float(targetData[24]) if len(targetData) > 24 else 0.0,
                "gain": float(targetData[25]) if len(targetData) > 25 else 0.0,
                "type": "UNKNOWN"
            })
            return targets
        except Exception as e:
            print(f"[parseTrackTLV] Single target parse failed: {e}")
            return []

    numDetectedTargets = int(tlvLength / targetSize)

    for i in range(numDetectedTargets):
        try:
            targetData = struct.unpack(targetStruct, tlvData[:targetSize])
            px, py, pz = float(targetData[1]), float(targetData[2]), float(targetData[3])
            vx, vy, vz = float(targetData[4]), float(targetData[5]), float(targetData[6])
            vals = (px, py, pz, vx, vy, vz)
            if not all(math.isfinite(v) for v in vals):
                tlvData = tlvData[targetSize:]; continue
            if _SAFE_MAX_VEL_MS and max(abs(vx), abs(vy), abs(vz)) > _SAFE_MAX_VEL_MS:
                tlvData = tlvData[targetSize:]; continue
            rng = float(np.sqrt(px*px + py*py + pz*pz))
            if _SAFE_MAX_RANGE_M and rng > _SAFE_MAX_RANGE_M:
                tlvData = tlvData[targetSize:]; continue
            vel_mag = float(np.sqrt(vx*vx + vy*vy + vz*vz))
            rng = float(np.sqrt(px*px + py*py + pz*pz))
            if rng > 1e-6:
                v_radial = float((vx*px + vy*py + vz*pz) / rng)
            else:
                v_radial = 0.0
            doppler = float((2.0 * v_radial) / _LAMBDA_M)
            snr = float(targetData[24]) if len(targetData) > 24 else 0.0
            gain = float(targetData[25]) if len(targetData) > 25 else 0.0
            signal = gain * snr if gain and snr else 0.0

            targets.append({
                "id": int(targetData[0]),
                "posX": float(targetData[1]),
                "posY": float(targetData[2]),
                "posZ": float(targetData[3]),
                "velX": float(targetData[4]),
                "velY": float(targetData[5]),
                "velZ": float(targetData[6]),
                "accX": float(targetData[7]),
                "accY": float(targetData[8]),
                "accZ": float(targetData[9]),
                "g": float(targetData[26]),
                "confidence": float(targetData[27]),
                "velocity": vel_mag,
                "radial_velocity": v_radial,
                "speed_kmh": float(vel_mag * 3.6),
                "distance": float(np.sqrt(targetData[1]**2 + targetData[2]**2 + targetData[3]**2)),
                "azimuth": float(np.degrees(np.arctan2(targetData[1], targetData[2] + 1e-6))),
                "elevation": float(np.degrees(np.arctan2(
                    targetData[3], np.sqrt(targetData[1]**2 + targetData[2]**2 + 1e-6)))),
                "snr": snr,
                "gain": gain,
                "doppler_frequency": doppler,
                "signal_level": signal,
                "type": "UNKNOWN"
            })
        except Exception as e:
            print(f"[parseTrackTLV] Object #{i} parse failed: {e}")
        tlvData = tlvData[targetSize:]

    return targets

def parseTrackHeightTLV(tlvData, tlvLength):
    targetStruct = 'I2f' #incoming data is an unsigned integer for TID, followed by 2 floats
    targetSize = struct.calcsize(targetStruct)
    numDetectedHeights = int(tlvLength/targetSize)
    heights = np.empty((numDetectedHeights,3))
    for i in range(numDetectedHeights):
        try:
            targetData = struct.unpack(targetStruct,tlvData[i * targetSize:(i + 1) * targetSize])
        except:
            print('ERROR: Target TLV parsing failed')
            return 0, heights

        heights[i,0] = targetData[0] # Target ID
        heights[i,1] = targetData[1] # maxZ
        heights[i,2] = targetData[2] # minZ

    return numDetectedHeights, heights

# Decode Target Index TLV
def parseTargetIndexTLV(tlvData, tlvLength):
    indexStruct = 'B' # One byte per index
    indexSize = struct.calcsize(indexStruct)
    numIndexes = int(tlvLength/indexSize)
    indexes = np.empty(numIndexes)
    for i in range(numIndexes):
        try:
            index = struct.unpack(indexStruct, tlvData[:indexSize])
        except:
            print('ERROR: Target Index TLV Parsing Failed')
            return indexes
        indexes[i] = int(index[0])
        tlvData = tlvData[indexSize:]

    return indexes

def parseVitalSignsTLV (tlvData, tlvLength):
    vitalsStruct = '2H33f'
    vitalsSize = struct.calcsize(vitalsStruct)
    
    # Initialize struct in case of error
    vitalsOutput = {}
    vitalsOutput ['id'] = 999
    vitalsOutput ['rangeBin'] = 0
    vitalsOutput ['breathDeviation'] = 0
    vitalsOutput ['heartRate'] = 0
    vitalsOutput ['breathRate'] = 0
    vitalsOutput ['heartWaveform'] = []
    vitalsOutput ['breathWaveform'] = []

    # Capture data for active patient
    try:
        vitalsData = struct.unpack(vitalsStruct, tlvData[:vitalsSize])
    except:
        print('ERROR: Vitals TLV Parsing Failed')
        return vitalsOutput
    
    # Parse this patient's data
    vitalsOutput ['id'] = vitalsData[0]
    vitalsOutput ['rangeBin'] = vitalsData[1]
    vitalsOutput ['breathDeviation'] = vitalsData[2]
    vitalsOutput ['heartRate'] = vitalsData[3]
    vitalsOutput ['breathRate'] = vitalsData [4]
    vitalsOutput ['heartWaveform'] = np.asarray(vitalsData[5:20])
    vitalsOutput ['breathWaveform'] = np.asarray(vitalsData[20:35])

    # Advance tlv data pointer to end of this TLV
    tlvData = tlvData[vitalsSize:]

    return vitalsOutput

def parseStatsTLV(tlvData):
    statStruct = '6I'  # 6 uint32_t values
    if len(tlvData) < struct.calcsize(statStruct):
        print("[parseStatsTLV] Invalid TLV length.")
        return {}

    stats = struct.unpack(statStruct, tlvData[:24])
    return {
        "interFrameProcTime": stats[0],
        "transmitOutputTime": stats[1],
        "interFrameMargin": stats[2],
        "interChirpMargin": stats[3],
        "activeFrameCPULoad": stats[4],
        "interFrameCPULoad": stats[5]
    }