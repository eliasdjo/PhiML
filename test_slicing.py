from phi.jax.flow import *

DOMAIN_V = dict(bounds=Box['x,y', 0:1, 0:1], x=31, y=31,
                      extrapolation=extrapolation.combine_sides(
                          x=extrapolation.ZERO,
                          y=(extrapolation.ZERO,
                             extrapolation.combine_by_direction(extrapolation.ZERO,
                                                                extrapolation.ConstantExtrapolation(-1)))))
velocity = CenteredGrid(tensor([0, 0], channel(vector='x, y')), **DOMAIN_V)
print(velocity.vector[0].extrapolation, '  -  ', velocity.vector[1].extrapolation)
print('done')
