# server.py
import socket

def getipaddrs(hostname):  # 只是为了显示IP，仅仅测试一下
    result = socket.getaddrinfo(hostname, None, 0, socket.SOCK_STREAM)
    # print(result)
    return [x[4][0] for x in result]

hostip_emu = '127.0.0.1'  #模拟
hostname = socket.gethostname() #真机
hostip_real = socket.gethostbyname(hostname+".local") #真机
print('host ip', hostip_real)
port = 6666  # Arbitrary non-privileged port
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(1)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
# s.bind((hostip_real[3], port)) #真机
s.bind((hostip_real, port)) #模拟1
s.listen(4)
i = 3
while True:
    try:
        conn, addr = s.accept()
        conn.settimeout(1)
        # cmd = "0,0,0,0\n"
        data = conn.recv(65536)
        if not data:
            print('no data')
        data = data.decode().split(',')
        # x速度，y速度，z速度，纬度，经度，高度，电池是否过低，是否飞行
        print(data)

        # 发送 eg：0,0,0,0
        # number1：-1~1：-1表示最大速度下降；1表示最大速度上升
        # number2：-1~1：-1表示最大速度向左yaw；1表示最大速度向右yaw
        # number3：-1~1：-1表示最大速度向前；1表示最大速度向后
        # number4：-1~1：-1表示最大速度向左；1表示最大速度向右
        cmd = input()
        if cmd == 'a':
            cmd = "0,0,0,1,1000\n"
        elif cmd == "w":
            cmd = "0,0,1,0,1000\n"
        elif cmd == "d":
            cmd = "0,0,0,-1,1000\n"
        else:
            cmd = "0,0,-1,0,1000\n"
    
        print(cmd)
        conn.sendall(cmd.encode())  # 数据发送
        # print(cmd.encode())
        conn.close()
    except:
        print('timeout')