import math 

'''
def xyz(cn,co):
    cnex=cn+1.390000
    cos=((co)*(co)-(cn)*(co)-(no)*(no))/(2*(cn)*(no))
    sins=1-(cos)*(cos)
    sin=math.sqrt(sins)

    oy=round((no)*(cos),6)+cnex
    oxp=round((no)*(sin),6)
    oxm=round(-(no)*(sin),6)

    xyz= f"""
        C        0.000000        1.390000        0.000000
        C        1.204362        0.695000        0.000000
        C        1.204362       -0.695000        0.000000
        C        0.000000       -1.390000        0.000000
        C       -1.204362       -0.695000        0.000000
        C       -1.204362        0.695000        0.000000
        N        0.000000        {cnex}          0.000000
        O        {oxp}           {oy}            0.000000
        O        {oxm}           {oy}            0.000000
        H        2.140000        1.235000        0.000000
        H        2.140000       -1.235000        0.000000
        H        0.000000       -2.470000        0.000000
        H       -2.140000       -1.235000        0.000000
        H       -2.140000        1.235000        0.000000
    """

    return xyz
'''


def xyz(ch,hh):
    ch2x=-ch/2
    ch2y=(math.sqrt(3))*(-ch2x)
    sin=(hh-ch2y)/(1.11)
    cos=math.sqrt(1-sin*sin)
    ch1x=(1.11)*(cos)
    ch1y=ch2y-hh
    ch2x=round(ch2x,6)
    ch2y=round(ch2y,6)
    ch1x=round(ch1x,6)
    ch1y=round(ch1y,6)


    xyz= f"""
        C        0.000000        0.000000        0.000000
        O        1.210000        0.000000        0.000000
        H        {ch2x}          {ch2y}          0.000000
        H        {ch1x}          {ch1y}          0.000000
       
    """

    return xyz
