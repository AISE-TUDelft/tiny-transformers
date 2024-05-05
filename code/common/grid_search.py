from __future__ import annotations
from itertools import product
from functools import reduce
from dataclasses import dataclass, field, Field
from typing import Dict, List, Any, Iterator

''' Copyright aral. This class is for 3.10 onwards as dataclasses decided to off themselves '''

@dataclass 
class Dimension: 
    ''' A dimension consists of a series of points 
        i.e. a list 
    '''
    points : List[Any | Dimension]

    def __iter__(self) -> Iterator[Any]:
        ''' iterate over this dimension, flattening sub-dimensions '''

        for point in self.points:
            if isinstance(point, Dimension): yield from iter(point)
            elif isinstance(point, Surface): yield from iter(point)
            else: yield point

    def __len__(self) -> int:
        ''' return size of all dimensions '''
        n_points = 0 
        for point in self.points: 
            if isinstance(point, Dimension): n_points += len(point)
            elif isinstance(point, Surface): n_points += len(point)
            else: n_points += 1
        return n_points 
    
    def __str__(self) -> str: 
        ''' prints points as list, or recurse into subdims '''
        nonsurface = ', '.join(
            f'*{point}' if isinstance(point, Dimension) else str(point)
            for point in filter(lambda x: not isinstance(x, Surface), self.points)) 
        surface = ','.join(str(point) for point in filter(lambda x: isinstance(x, Surface), self.points))
        
        if len(nonsurface) > 0 and len(surface) > 0:
            return nonsurface + ', ' + surface 
        elif len(nonsurface) > 0: 
            return nonsurface 
        else: 
            return surface 

def search(*points) -> Field: 
    ''' to be used within a Surface definition '''
    return field(default_factory=lambda: Dimension(points))

@dataclass 
class Surface: 
    ''' A surface consists of at least two dimensions 
        i.e. multiple lists, and optional static points 
    '''

@dataclass 
class GridSearch(Surface): 
    ''' Grid search a dataclass. Provide searchable values as follows: 
        ```python
        @dataclass
        class Params(GridSearch):
            a = search(1,2,3)
        
        for param in Params(): ...
        ```
    '''

    def __post_init__(self): 
        # leaving this method here as it may be useful later again
        pass

    def __str__(self) -> str: 

        string = f'\n\033[95m{len(self):3}\033[0m \033[1m{self.__class__.__name__}:\033[0m'

        max_k_len = max(map(len, self.__dict__.keys()))
        for k,v in self.__dict__.items(): 
            # in case of a multiline value (from a Surface), we pad the block
            v_string = '\n    '.join(str(v).splitlines()) 
            n_combinations = f'\033[95m{len(v) if hasattr(v, "__len__") else 1}\033[0m'
            # string += f'\n  {n} \033[1m{k:s}\033[0m: [{v_string}]'
            string += '\n  {} \033[3m{:{}s}\033[0m: [{}]'.format(
                n_combinations, k, max_k_len, v_string
            )

        return string + '\n'

    def __len__(self) -> int:
        return reduce(
                lambda a,b: a*b, 
                map(
                    len,
                    (d for d in self.__dimensions.values()) 
            ), 1)

    def __iter__(self) -> Iterator[Surface]:
        keys = self.__dimensions.keys()
        for i, instance in enumerate(product(*[v for v in self.__dimensions.values()])):
            point = self.__class__(**self.__static_points, **dict(zip(keys, instance)))
            
            point.__class__.__name__ += f'-{i}' \
                if not point.__class__.__name__.__contains__('-') else f'\b{i}'
            yield point

    @property 
    def __static_points(self) -> Dict[str, Any] :
        ''' dictionary of { var_one: Any, var_two: Any, ... } '''
        return {k:v for k,v in self.__dict__.items() if not isinstance(v, Dimension)}

    @property 
    def __dimensions(self) -> Dict[str, Dimension]:
        ''' dictionary of { var_one: Dimension, var_two: Dimension, ... } '''
        return {k:v for k,v in self.__dict__.items() if isinstance(v,Dimension)}

