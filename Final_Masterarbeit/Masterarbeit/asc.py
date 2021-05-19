import time
import busio
import digitalio
import board
import adafruit_mcp3xxx.mcp3008 as MCP
from adafruit_mcp3xxx.analog_in import AnalogIn
import matplotlib.pyplot as plt
import numpy as np
import datetime
 
 
# open a binary file in write mode
# file = open("signa1", "wb")
# # save array to the file
# np.save(file, arr)
# # close the file
# file.close
# 
# # open the file in read binary mode
# file = open("arr", "rb")
# #read the file to numpy array
# arr1 = np.load(file)
# #close the file
# file.close

# create the spi bus
spi = busio.SPI(clock=board.SCK, MISO=board.MISO, MOSI=board.MOSI)
 
# create the cs (chip select)
cs1 = digitalio.DigitalInOut(board.D22)
cs2 = digitalio.DigitalInOut(board.D27)
 
# create the mcp object
mcp1 = MCP.MCP3008(spi, cs1)
mcp2 = MCP.MCP3008(spi, cs2)
 
# create an analog input channel on pin 0
chan0 = AnalogIn(mcp1, MCP.P0)
chan1 = AnalogIn(mcp1, MCP.P1)
chan2 = AnalogIn(mcp1, MCP.P2)
chan3 = AnalogIn(mcp1, MCP.P3)
chan4 = AnalogIn(mcp1, MCP.P4)
chan5 = AnalogIn(mcp1, MCP.P5)
chan6 = AnalogIn(mcp1, MCP.P6)
chan7 = AnalogIn(mcp1, MCP.P7)
chan8 = AnalogIn(mcp2, MCP.P0)
chan9 = AnalogIn(mcp2, MCP.P1)
chan10 = AnalogIn(mcp2, MCP.P2)
chan11 = AnalogIn(mcp2, MCP.P3)
chan12 = AnalogIn(mcp2, MCP.P4)
chan13 = AnalogIn(mcp2, MCP.P5)
chan14 = AnalogIn(mcp2, MCP.P6)
chan15 = AnalogIn(mcp2, MCP.P7)

t = 0
t_array = []

voltage_Signal = [[] for i in range(16)]

print('Start :', datetime.datetime.now())

while True:
    
    print('ADC Voltage: ', "%.6f" % chan7.voltage)
    
    for i in range(16):
        exec("voltage_Signal["+str(i)+"]" + "+=" + "[chan"+str(i)+".voltage]")
        
#     voltage_Signal[0] += [chan0.voltage]
#     voltage_Signal[1] += [chan1.voltage]
#     voltage_Signal[2] += [chan2.voltage]
#     voltage_Signal[3] += [chan3.voltage]
#     voltage_Signal[4] += [chan4.voltage]
#     voltage_Signal[5] += [chan5.voltage]
#     voltage_Signal[6] += [chan6.voltage]
#     voltage_Signal[7] += [chan7.voltage]
#     voltage_Signal[8] += [chan8.voltage]
#     voltage_Signal[9] += [chan9.voltage]
#     voltage_Signal[10] += [chan10.voltage]
#     voltage_Signal[11] += [chan11.voltage]
#     voltage_Signal[12] += [chan12.voltage]
#     voltage_Signal[13] += [chan13.voltage]
#     voltage_Signal[14] += [chan14.voltage]
#     voltage_Signal[15] += [chan15.voltage]
    
#     time.sleep(0.1)
    
    t_array += [t/33]
    
    t += 1
    
    if t == 660:
        break
    

print('End :', datetime.datetime.now())

# for i in range(16):
#     
#     #Save signal as an array
#     # open a binary file in write mode
#     file = open("/media/pi/Volume/Masterarbeit/Test_reverse_splitter_signal7/signal_silver_"+str(i), "wb")
#     # save array to the file
#     voltage_Signal[i] = np.array(voltage_Signal[i])
#     np.save(file, voltage_Signal[i])
#     # close the file
#     file.close()
# 
#     # Plot signal
#     fig, ax = plt.subplots()
#     
#     ax.set_xlabel('Zeit (s)')
#     ax.set_ylabel('Spannungswert (V)')
#     fig.suptitle('Signal '+str(i))
#     ax.plot(t_array,voltage_Signal[i])
# 
#     fig.savefig("/media/pi/Volume/Masterarbeit/Test_reverse_splitter_signal7/signal_silver_"+str(i)+".png",orientation='landscape')
#     fig.clear()
    
#print(voltage_Signal)