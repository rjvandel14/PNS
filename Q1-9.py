A = 1.2 

lambda_voice = 25 * A          
lambda_total_vc = 0.8 * A     
lambda_low = 0.8 * lambda_total_vc   
lambda_high = 0.2 * lambda_total_vc  

d_voice = 5 / 60.0             
d_vc = 18 / 60.0               

rho_voice = lambda_voice * d_voice
rho_low = lambda_low * d_vc
rho_high = lambda_high * d_vc

rho = [rho_voice, rho_low, rho_high]  
b = [1, 2, 3]                       
C = 4                               

# Kaufman–Roberts recursion
q = [0.0] * (C + 1)
q[0] = 1.0 

for c in range(1, C + 1):
    s = 0.0
    for k in range(3):        
        if b[k] <= c:
            s += rho[k] * b[k] * q[c - b[k]]
    q[c] = s / c              

# Normalise
G = sum(q)
q = [qc / G for qc in q]

# Blocking probabilities
B_voice = sum(q[C - b[0] + 1:])  
B_low   = sum(q[C - b[1] + 1:])  
B_high  = sum(q[C - b[2] + 1:])  

print(f"B_voice = {B_voice:.5f}")
print(f"B_low   = {B_low:.5f}")
print(f"B_high  = {B_high:.5f}")
