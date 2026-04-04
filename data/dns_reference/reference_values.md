# DNS/Experimental Reference Values for Paper Validation

## Cylinder Re=100
- **Cd = 1.35** (time-averaged, multiple DNS: Henderson 1995, Park 1998, Williamson 1996)
- **St = 0.164** (vortex shedding Strouhal number)
- **Cl_rms ≈ 0.23**
- **Separation angle ≈ 117°**
- Note: unsteady (vortex shedding). Compare time-averaged Cd and St.

## Periodic Hills Re=5600
- **Separation: x/H = 0.169** (Krank, Kronbichler, Wall 2018, 8th-order DG DNS)
- **Reattachment: x/H = 5.036** (Krank et al. 2018)
- **Cf_min = -0.004637** at x/H = 2.656
- **Cf_max = 0.038467** at x/H = 8.641
- **Secondary bubble**: separation 7.03, reattachment 7.31
- Earlier reference: Breuer et al. (2009): sep 0.22, reattach 4.72
- DATA DOWNLOADED: `data/dns_reference/hills/KKW_DNS_*.dat`
  - Cf(x) and Cp(x) along bottom wall (577 points)
  - Velocity profiles at x/H = 0.05, 0.5, 1, 2, 3, 4, 5, 6, 7, 8
  - Full 2D mean + RMS fields (131K points each)
- Source: DOI 10.14459/2018mp1415670

## Square Duct Re_b=3500
- **f = 0.0419** (friction factor, Pinelli et al. 2010, JFM 644)
- **U_cl/U_b ≈ 1.43** (centerline to bulk velocity ratio)
- **Secondary flow magnitude: 1-2% of U_b**
- Also: Gavrilakis (1992) JFM 244 at Re_tau=150
- Need: U(y,z) contours, V/W secondary flow pattern, wall shear

## Sphere Re=200
- **Cd = 0.775** (Johnson & Patel 1999, JFM 378, Table 2)
- **Separation angle ≈ 153°** from front stagnation
- **Wake length ≈ 1.5D**
- Flow is steady and axisymmetric at Re=200 (unsteady onset ≈ Re=270)
- Clift, Grace & Weber (1978) correlation gives Cd=0.77, consistent
- Also: Tomboulides & Orszag (2000) JCP 164
