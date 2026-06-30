import sisl
import matplotlib.pyplot as plt

def build_geom(L):
    # 1 antraceno = 2 unit cells
    uc = sisl.geom.agnr(width=7, bond=1.42)
    geom = uc.tile(2 * L, axis=0)
    geom.set_nsc([1, 1, 1])
    
    print(f"L={L}, initial atoms: {geom.na}")
    
    # Prune dangling bonds iterativamente
    while True:
        to_remove = []
        for ia in geom:
            idx = geom.close(ia, R=1.5) # Devuelve array 1D que incluye a sí mismo
            if len(idx) == 2: # 1 a sí mismo + 1 vecino = coordinación 1
                to_remove.append(ia)
        
        if not to_remove:
            break
        print(f"Removing {len(to_remove)} dangling bonds: {to_remove}")
        geom = geom.remove(to_remove)
        
    print(f"L={L}, final atoms: {geom.na}")
    return geom

geom = build_geom(4)

# Plot geometry to verify
plt.figure(figsize=(10, 4))
xyz = geom.xyz
plt.scatter(xyz[:, 0], xyz[:, 1], c='k', s=50)
for ia in geom:
    idx = geom.close(ia, R=1.5)
    for j in idx:
        if j > ia:
            plt.plot([xyz[ia, 0], xyz[j, 0]], [xyz[ia, 1], xyz[j, 1]], 'k-', lw=1.5)
plt.axis('equal')
plt.title('7-AGNR L=4 (Pruned)')
plt.savefig('test_geom_L4.png')
print("Saved geometry plot to test_geom_L4.png")
