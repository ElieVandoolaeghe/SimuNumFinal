import numpy as np
from matplotlib import cm
from matplotlib.collections import EllipseCollection
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pickle
            
class CollisionEvent:
    """
    Object contains all information about a collision event
    which are necessary to update the velocity after the collision.
    For MD of hard spheres (with hard bond-length dimer interactions)
    in a rectangular simulation box with hard walls, there are only
    two distinct collision types:
    1) wall collision of particle i with vertical or horizontal wall
    2) external (or dimer bond-length) collision between particle i and j
    """
    def __init__(self, Type = 'wall or other', dt = np.inf, mono_1 = 0, mono_2 = 0, w_dir = 1):
        """
        Type = 'wall' or other
        dt = remaining time until collision
        mono_1 = index of monomer
        mono_2 = if inter-particle collision, index of second monomer
        w_dir = if wall collision, direction of wall
        (   w_dir = 0 if wall in x direction, i.e. vertical walls
            w_dir = 1 if wall in y direction, i.e. horizontal walls   )
        """
        self.Type = Type
        self.dt = dt
        self.mono_1 = mono_1
        self.mono_2 = mono_2  # only importent for interparticle collisions
        self.w_dir = w_dir # only important for wall collisions
        
        
    def __str__(self):
        if self.Type == 'wall':
            return "Event type: {:s}, dt: {:.8f}, p1 = {:d}, dim = {:d}".format(self.Type, self.dt, self.mono_1, self.w_dir)
        else:
            return "Event type: {:s}, dt: {:.8f}, p1 = {:d}, p2 = {:d}".format(self.Type, self.dt, self.mono_1, self.mono_2)

class Monomers:
    """
    Class for event-driven molecular dynamics simulation of hard spheres:
    -Object contains all information about a two-dimensional monomer system
    of hard spheres confined in a rectengular box of hard walls.
    -A configuration is fully described by the simulation box and
    the particles positions, velocities, radiai, and masses.
    -Initial configuration of $N$ monomers has random positions (without overlap)
    and velocities of random orientation and norms satisfying
    $E = \sum_i^N m_i / 2 (v_i)^2 = N d/2 k_B T$, with $d$ being the dimension,
    $k_B$ the Boltzmann constant, and $T$ the temperature.
    -Class contains all functions for an event-driven molecular dynamics (MD)
    simulation. Essentail for inter-particle collsions is the mono_pair array,
    which book-keeps all combinations without repetition of particle-index
    pairs for inter-particles collisions, e.g. for $N = 3$ particles
    indices = 0, 1, 2
    mono_pair = [[0,1], [0,2], [1,2]]
    -Monomers can be initialized with individual radiai and density = mass/volume.
    For example:
    NumberOfMonomers = 7
    NumberMono_per_kind = [ 2, 5]
    Radiai_per_kind = [ 0.2, 0.5]
    Densities_per_kind = [ 2.2, 5.5]
    then monomers mono_0, mono_1 have radius 0.2 and mass 2.2*pi*0.2^2
    and monomers mono_2,...,mono_6 have radius 0.5 and mass 5.5*pi*0.5^2
    -The dictionary of this class can be saved in a pickle file at any time of
    the MD simulation. If the filename of this dictionary is passed to
    __init__ the simulation can be continued at any point in the future.
    IMPORTANT! If system is initialized from file, then other parameters
    of __init__ are ignored!
    """
    def __init__(self, NumberOfMonomers = 4, L_xMin = 0, L_xMax = 1, L_yMin = 0, L_yMax = 1, NumberMono_per_kind = np.array([4]), Radiai_per_kind = 0.5*np.ones(1), Densities_per_kind = np.ones(1), k_BT = 1, FilePath = './Configuration.p'):
        try:
            self.__dict__ = pickle.load( open( FilePath, "rb" ) )
            print("IMPORTANT! System is initialized from file %s, i.e. other input parameters of __init__ are ignored!" % FilePath)
        except:
            assert ( NumberOfMonomers > 0 )
            assert ( (L_xMin < L_xMax) and (L_yMin < L_yMax) )
            self.NM = NumberOfMonomers
            self.DIM = 2 #dimension of system
            self.BoxLimMin = np.array([ L_xMin, L_yMin])
            self.BoxLimMax = np.array([ L_xMax, L_yMax])
            self.mass =-1* np.ones( self.NM ) # Masses, negative mass means not initialized
            self.rad =-1* np.ones( self.NM ) # Radiai, negative radiai means not initialized
            self.pos = np.empty( (self.NM, self.DIM) ) # Positions, not initalized but desired shape
            self.vel = np.empty( (self.NM, self.DIM) ) # Velocities, not initalized but desired shape
            self.mono_pairs = np.array( [ (k,l) for k in range(self.NM) for l in range( k+1,self.NM ) ] )
            self.next_wall_coll = CollisionEvent( 'wall', np.inf, 0, 0, 0)
            self.next_mono_coll = CollisionEvent( 'mono', np.inf, 0, 0, 0)
        
            self.assignRadiaiMassesVelocities(NumberMono_per_kind, Radiai_per_kind, Densities_per_kind, k_BT )
            self.assignRandomMonoPos( )
    
    def save_configuration(self, FilePath = 'MonomerConfiguration.p'):
        '''Saves configuration. Callable at any time during simulation.'''
        #print( self.__dict__ )
    
    def assignRadiaiMassesVelocities(self, NumberMono_per_kind = np.array([4]), Radiai_per_kind = 0.5*np.ones(1), Densities_per_kind = np.ones(1), k_BT = 1 ):
        '''
        Make this a PRIVATE function -> cannot be called outside class definition.
        '''
        '''initialize radiai and masses'''
        assert( sum(NumberMono_per_kind) == self.NM )
        assert( isinstance(Radiai_per_kind,np.ndarray) and (Radiai_per_kind.ndim == 1) )
        assert( (Radiai_per_kind.shape == NumberMono_per_kind.shape) and (Radiai_per_kind.shape == Densities_per_kind.shape))
        l = 0
        for i in range(len(NumberMono_per_kind)):
            for j in range(NumberMono_per_kind[i]):
                self.rad[l+j] = Radiai_per_kind[i]
                self.mass[l+j] = Densities_per_kind[i]*np.pi*Radiai_per_kind[i]
            l += NumberMono_per_kind[i]
        
        '''initialize velocities'''
        assert( k_BT > 0 )
        self.vel = np.random.rand(self.NM,self.DIM)
        sums = np.sum(self.mass/2*(self.vel[:,0]**2+self.vel[:,1]**2))
        gamma = self.NM*self.DIM*k_BT/sums
        self.vel *= np.ones(np.shape(self.vel))*np.sqrt(gamma)
    
        
    
    def assignRandomMonoPos(self, start_index = 0 ):
        '''
        Make this a PRIVATE function -> cannot be called outside class definition.
        Initialize random positions without overlap between monomers and wall.
        '''
        assert( min(self.rad) > 0 )
        Mono_Num, infiniteLoopTest = start_index, 0
        BoxLength = self.BoxLimMax - self.BoxLimMin
        while Mono_Num < self.NM and infiniteLoopTest < 10**4:
            infiniteLoopTest += 1
            self.pos[Mono_Num,:] = np.random.rand(1,2)*self.BoxLimMax
            while np.any(self.BoxLimMax-self.pos[Mono_Num,:]-self.rad[Mono_Num] < 0) | np.any(self.pos[Mono_Num,:]-self.BoxLimMin-self.rad[Mono_Num] < 0) | np.any((abs(self.pos[0:Mono_Num,:]-self.pos[Mono_Num,:])) - self.rad[Mono_Num] - self.rad[0:Mono_Num,None] < 0):
                self.pos[Mono_Num,:] = np.random.rand(1,2)*self.BoxLimMax
            Mono_Num += 1

    
    def __str__(self, index = 'all'):
        if index == 'all':
            return "\nMonomers with:\nposition = " + str(self.pos) + "\nvelocity = " + str(self.vel) + "\nradius = " + str(self.rad) + "\nmass = " + str(self.mass)
        else:
            return "\nMonomer at index = " + str(index) + " with:\nposition = " + str(self.pos[index]) + "\nvelocity = " + str(self.vel[index]) + "\nradius = " + str(self.rad[index]) + "\nmass = " + str(self.mass[index])
        
    def Wall_time(self):
        '''
        -Function computes list of remaining time dt until future
        wall collision in x and y direction for every particle.
        Then, it stores collision parameters of the event with
        the smallest dt in the object next_wall_coll.
        -Meaning of future:
        if v > 0: solve BoxLimMax - rad = x + v * dt
        else:     solve BoxLimMin + rad = x + v * dt
        '''
        
        #---> Your turn!
        collDist = np.where(self.vel > 0, self.BoxLimMax - self.rad[:,None], self.BoxLimMin + self.rad[:,None])
        dt_List = (collDist - self.pos) / self.vel
        index = np.argmin(dt_List)
        minCollTime = np.amin(dt_List)
        collision_disk = index //2
        wall_direction = 0 if index %2 == 0 else 1

        self.next_wall_coll.dt = minCollTime
        self.next_wall_coll.mono_1 = collision_disk
        #self.next_wall_coll.mono_2 = not necessary
        self.next_wall_coll.w_dir = wall_direction
        
        
    def Mono_pair_time(self):
        '''
        - Function computes list of remaining time dt until
        future external collition between all combinations of
        monomer pairs without repetition. Then, it stores
        collision parameters of the event with
        the smallest dt in the object next_mono_coll.
        - If particles move away from each other, i.e.
        scal >= 0 or Omega < 0, then remaining dt is infinity.
        '''
        mono_i = self.mono_pairs[:,0] # List of collision partner 1
        mono_j = self.mono_pairs[:,1] # List of collision partner 2
        
        #print('mono_i = ', mono_i)
        dx_dy = self.pos[mono_i]-self.pos[mono_j]
        dvx_dvy = self.vel[mono_i]-self.vel[mono_j]
        dist = np.linalg.norm(dx_dy, axis = 1)
        dtmin = self.rad[mono_i] + self.rad[mono_j]
        a = np.square(np.linalg.norm(dvx_dvy, axis=1))
        b = 2*(np.multiply(dvx_dvy[:,0],dx_dy[:,0])+np.multiply(dvx_dvy[:,1],dx_dy[:,1]))
        c = np.square(dist)-np.square(dtmin)
        
        omega = np.square(b) - 4*np.multiply(a,c)
        delta_minus = []
        for i in range(len(omega)):
            if omega[i] > 0 and b[i]<0:
                delta_minus.append((-b[i] - np.sqrt(omega[i]))/(2*a[i]))
            else :
                delta_minus.append(np.inf)

        k = np.argmin(delta_minus)
        minCollTime = np.amin(delta_minus)
        
        collision_disk_1 = mono_i[k]
        collision_disk_2 = mono_j[k]
        
        self.next_mono_coll.dt = minCollTime
        self.next_mono_coll.mono_1 = collision_disk_1
        self.next_mono_coll.mono_2 = collision_disk_2
        #self.next_mono_coll.w_dir = not necessary
        
    def compute_next_event(self):
        '''
        Function gets event information about:
        1) next possible wall event
        2) next possible pair event
        Function returns event info of event with
        minimal time, i.e. the clostest in future.
        '''
        
        
        self.Wall_time()
        self.Mono_pair_time()
        if self.next_wall_coll.dt < self.next_mono_coll.dt :
            return self.next_wall_coll
        else:
            return self.next_mono_coll

    
    def compute_new_velocities(self, next_event):
        '''
        Function updates the velocities of the monomer(s)
        involved in collision event.
        Update depends on event type.
        Ellastic wall collisions in x direction reverse vx.
        Ellastic pair collisions follow: https://en.wikipedia.org/wiki/Elastic_collision#Two-dimensional_collision_with_two_moving_objects
        '''
        if next_event.Type == 'wall' :
            if self.next_wall_coll.w_dir == 0 :
                self.vel[self.next_wall_coll.mono_1, 0] *=  -1
            else :
                self.vel[self.next_wall_coll.mono_1, 1] *=  -1
        elif next_event.Type == 'mono' :
                dist_till_coll = self.pos[next_event.mono_1] - self.pos[next_event.mono_2]
                vect_unit = dist_till_coll * (1/np.linalg.norm(dist_till_coll))
                speed_diff = self.vel[next_event.mono_1] - self.vel[next_event.mono_2]

                self.vel[next_event.mono_1] -=(2*(self.mass[next_event.mono_2])/(self.mass[next_event.mono_2]+self.mass[next_event.mono_1])) * np.dot(dist_till_coll,speed_diff)* dist_till_coll
                
                self.vel[next_event.mono_2] +=(2*(self.mass[next_event.mono_1])/(self.mass[next_event.mono_2]+self.mass[next_event.mono_1])) * np.dot(dist_till_coll,speed_diff)* dist_till_coll
        
                        

                

    def snapshot(self, FileName = './snapshot.png', Title = '$t = $?'):
        '''
        Function saves a snapshot of current configuration,
        i.e. particle positions as circles of corresponding radius,
        velocities as arrows on particles,
        blue dashed lines for the hard walls of the simulation box.
        '''
        fig, ax = plt.subplots( dpi=300 )
        L_xMin, L_xMax = self.BoxLimMin[0], self.BoxLimMax[0]
        L_yMin, L_yMax = self.BoxLimMin[1], self.BoxLimMax[1]
        BorderGap = 0.1*(L_xMax - L_xMin)
        ax.set_xlim(L_xMin-BorderGap, L_xMax+BorderGap)
        ax.set_ylim(L_yMin-BorderGap, L_yMax+BorderGap)

        #--->plot hard walls (rectangle)
        rect = mpatches.Rectangle((L_xMin,L_yMin), L_xMax-L_xMin, L_yMax-L_yMin, linestyle='dashed', ec='gray', fc='None')
        ax.add_patch(rect)
        ax.set_aspect('equal')
        ax.set_xlabel('$x$ position')
        ax.set_ylabel('$y$ position')
        
        #--->plot monomer positions as circles
        MonomerColors = np.linspace( 0.2, 0.95, self.NM)
        Width, Hight, Angle = 2*self.rad, 2*self.rad, np.zeros( self.NM )
        collection = EllipseCollection( Width, Hight, Angle, units='x', offsets=self.pos,
                       transOffset=ax.transData, cmap='nipy_spectral', edgecolor = 'k')
        collection.set_array(MonomerColors)
        collection.set_clim(0, 1) # <--- we set the limit for the color code
        ax.add_collection(collection)

        #--->plot velocities as arrows
        ax.quiver( self.pos[:,0], self.pos[:,1], self.vel[:,0], self.vel[:,1] , units = 'dots', scale_units = 'dots')
        
        plt.title(Title)
        plt.savefig(FileName)
        plt.close()

        
class Dimers(Monomers):
    """
    --> Class derived from Monomers.
    --> See also comments in Monomer class.
    --> Class for event-driven molecular dynamics simulation of hard-sphere
    system with DIMERS (and monomers). Two hard-sphere monomers form a dimer,
    and experience additional ellastic collisions at the maximum
    bond length of the dimer. The bond length is defined in units of the
    minimal distance of the monomers, i.e. the sum of their radiai.
    -Next to the monomer information, the maximum dimer bond length is needed
    to fully describe one configuration.
    -Initial configuration of $N$ monomers has random positions without overlap
    and separation of dimer pairs is smaller than the bond length.
    Velocities have random orientations and norms that satisfy
    $E = \sum_i^N m_i / 2 (v_i)^2 = N d/2 k_B T$, with $d$ being the dimension,
    $k_B$ the Boltzmann constant, and $T$ the temperature.
    -Class contains all functions for an event-driven molecular dynamics (MD)
    simulation. Essentail for all inter-particle collsions is the mono_pair array
    (explained in momonmer class). Essentail for the ellastic bond collision
    of the dimers is the dimer_pair array which book-keeps index pairs of
    monomers that form a dimer. For example, for a system of $N = 10$ monomers
    and $M = 2$ dimers:
    monomer indices = 0, 1, 2, 3, ..., 9
    dimer_pair = [[0,2], [1,3]]
    -Monomers can be initialized with individual radiai and density = mass/volume.
    For example:
    NumberOfMonomers = 10
    NumberOfDimers = 2
    bond_length_scale = 1.2
    NumberMono_per_kind = [ 2, 2, 6]
    Radiai_per_kind = [ 0.2, 0.5, 0.1]
    Densities_per_kind = [ 2.2, 5.5, 1.1]
    then monomers mono_0, mono_1 have radius 0.2 and mass 2.2*pi*0.2^2
    and monomers mono_2, mono_3 have radius 0.5 and mass 5.5*pi*0.5^2
    and monomers mono_4,..., mono_9 have radius 0.1 and mass 1.1*pi*0.1^2
    dimer pairs are: (mono_0, mono_2), (mono_1, mono_3) with bond length 1.2*(0.2+0.5)
    see bond_length_scale and radiai
    -The dictionary of this class can be saved in a pickle file at any time of
    the MD simulation. If the filename of this dictionary is passed to
    __init__ the simulation can be continued at any point in the future.
    IMPORTANT! If system is initialized from file, then other parameters
    of __init__ are ignored!
    """
    def __init__(self, NumberOfMonomers = 4, NumberOfDimers = 2, L_xMin = 0, L_xMax = 1, L_yMin = 0, L_yMax = 1, NumberMono_per_kind = np.array([4]), Radiai_per_kind = 0.5*np.ones(1), Densities_per_kind = np.ones(1), bond_length_scale = 1.2, k_BT = 1, FilePath = './Configuration.p'):
        #if __init__() defined in derived class -> child does NOT inherit parent's __init__()
        try:
            self.__dict__ = pickle.load( open( FilePath, "rb" ) )
            print("IMPORTANT! System is initialized from file %s, i.e. other input parameters of __init__ are ignored!" % FilePath)
        except:
            assert ( (NumberOfDimers > 0) and (NumberOfMonomers >= 2*NumberOfDimers) )
            assert ( bond_length_scale > 1. ) # is in units of minimal distance of respective monomer pair
            Monomers.__init__(self, NumberOfMonomers, L_xMin, L_xMax, L_yMin, L_yMax, NumberMono_per_kind, Radiai_per_kind, Densities_per_kind, k_BT )
            self.ND = NumberOfDimers
            self.dimer_pairs = np.array([[k,self.ND+k] for k in range(self.ND)])#choice 2 -> more practical than [2*k,2*k+1]
            mono_i, mono_j = self.dimer_pairs[:,0], self.dimer_pairs[:,1]
            self.bond_length = bond_length_scale * ( self.rad[mono_i] + self.rad[mono_j] )
            self.next_dimer_coll = CollisionEvent( 'dimer', 0, 0, 0, 0)
            
            '''
            Positions initialized as pure monomer system by monomer __init__.
            ---> Reinitalize all monomer positions, but place dimer pairs first
            while respecting the maximal distance given by the bond length!
            '''
            self.assignRandomDimerPos()
            self.assignRandomMonoPos( 2*NumberOfDimers )
    
    def assignRandomDimerPos(self):
        '''
        Make this is a PRIVATE function -> cannot be called outside class definition
        initialize random positions without overlap between monomers and wall
        '''
        dimer_new_index, infiniteLoopTest = 0, 0
        BoxLength = self.BoxLimMax - self.BoxLimMin
        while dimer_new_index < self.ND and infiniteLoopTest < 10**4:
            infiniteLoopTest += 1
            mono_i, mono_j = dimer_new = self.dimer_pairs[dimer_new_index]
            # Your turn to place the dimers one after another such that
            # there are no overlaps between monomers and hard walls and
            # dimer pairs are not further appart than their max bond length.
        if dimer_new_index != self.ND:
            print('Failed to initialize all dimer positions.\nIncrease simulation box size!')
            exit()
        
        
    def __str__(self, index = 'all'):
        if index == 'all':
            return Monomers.__str__(self) + "\ndimer pairs = " + str(self.dimer_pairs) + "\nwith max bond length = " + str(self.bond_length)
        else:
            return "\nDimer pair " + str(index) + " consists of monomers = " + str(self.dimer_pairs[index]) + "\nwith max bond length = " + str(self.bond_length[index]) + Monomers.__str__(self, self.dimer_pairs[index][0]) + Monomers.__str__(self, self.dimer_pairs[index][1])

    def Dimer_pair_time(self):
        '''
        Function computes list of remaining time dt until
        future dimer bond collition for all dimer pairs.
        Then, it stores collision parameters of the event with
        the smallest dt in the object next_dimer_coll.
        '''
        mono_i = self.dimer_pairs[:,0] # List of collision partner 1
        mono_j = self.dimer_pairs[:,1] # List of collision partner 2
        delta_x = self.pos[mono_i,0] - self.pos[mono_j,0]
        delta_y = self.pos[mono_i,1] - self.pos[mono_j,1]
        delta_vx = self.vel[mono_i,0] - self.vel[mono_j,0]
        delta_vy = self.vel[mono_i,1] - self.vel[mono_j,1]
        dist = np.sqrt(delta_x**2+delta_y**2)
        a = delta_vx**2 + delta_vy**2
        b = 2 *(delta_vx*delta_x + delta_vy * delta_y)
        c = np.minimum(0,delta_x**2+delta_y**2 - self.bond_length**2)
        #deltaplus = (1/(2*a))*(-b + np.sqrt(b**2 - 4*a*c))
        dt_list = []
        for i in range(len(a)) :
            deltaplus = (1/(2*a[i]))*(-b[i] + np.sqrt(b[i]**2 - 4*a[i]*c[i]))
            if deltaplus > 0 :
                dt_list.append(deltaplus)
            else :
                dt_list.append(np.inf)
        
        index = np.argmin(dt_list)
        minCollTime = np.amin(dt_list)
        collision_disk_1 = mono_i[index]
        collision_disk_2 = mono_j[index]
        
        self.next_dimer_coll.dt = minCollTime
        self.next_dimer_coll.mono_1 = collision_disk_1
        self.next_dimer_coll.mono_2 = collision_disk_2
        print('Dimer_pair', minCollTime, collision_disk_1,collision_disk_2)
        #self.next_dimer_coll.w_dir = not necessary


    def compute_next_event(self):
        '''
        Function gets event information about:
        1) next possible wall event
        2) next possible pair event
        Function returns event info of event with
        minimal time, i.e. the clostest in future.
        '''
            
        self.Wall_time()
        self.Mono_pair_time()
        self.Dimer_pair_time()
        if self.next_wall_coll.dt < self.next_mono_coll.dt and self.next_wall_coll.dt < self.next_dimer_coll.dt :
            return self.next_wall_coll
        elif self.next_mono_coll.dt < self.next_dimer_coll.dt :
            return self.next_mono_coll
        else :
            return self.next_dimer_coll

    def compute_new_velocities(self, next_event):
        '''
        Function updates the velocities of the monomer(s)
        involved in collision event.
        Update depends on event type.
        Ellastic wall collisions in x direction reverse vx.
        Ellastic pair collisions follow: https://en.wikipedia.org/wiki/Elastic_collision#Two dimensional_collision_with_two_moving_objects'''
        if next_event.Type == 'wall' :
            if self.next_wall_coll.w_dir == 0 :
                self.vel[self.next_wall_coll.mono_1, 0] *=  -1
            else :
                self.vel[self.next_wall_coll.mono_1, 1] *=  -1
        elif next_event.Type == 'mono' :
            dist_till_coll = self.pos[self.next_mono_coll.mono_1] - self.pos[self.next_mono_coll.mono_2]
            vect_unit = dist_till_coll * (1/np.linalg.norm(dist_till_coll))
            speed_diff = self.vel[self.next_mono_coll.mono_1] - self.vel[self.next_mono_coll.mono_2]
    
            self.vel[self.next_mono_coll.mono_1] -=(2*(self.mass[self.next_mono_coll.mono_2])/(self.mass[self.next_mono_coll.mono_2]+self.mass[self.next_mono_coll.mono_1])) * np.dot(dist_till_coll,speed_diff)* dist_till_coll
            self.vel[self.next_mono_coll.mono_2] +=(2*(self.mass[self.next_mono_coll.mono_1])/(self.mass[self.next_mono_coll.mono_2]+self.mass[self.next_mono_coll.mono_1])) * np.dot(dist_till_coll,speed_diff)* dist_till_coll

        
        elif next_event.Type == 'dimer' :
            dist_till_coll = self.pos[self.next_dimer_coll.mono_1] - self.pos[self.next_dimer_coll.mono_2]
            vect_unit = dist_till_coll * (1/np.linalg.norm(dist_till_coll))
            speed_diff = self.vel[self.next_dimer_coll.mono_1] - self.vel[self.next_dimer_coll.mono_2]
            
            self.vel[self.next_dimer_coll.mono_1] -=(2*(self.mass[self.next_dimer_coll.mono_2])/(self.mass[self.next_dimer_coll.mono_2]+self.mass[self.next_dimer_coll.mono_1])) * np.dot(dist_till_coll,speed_diff)* dist_till_coll
            self.vel[self.next_dimer_coll.mono_2] +=(2*(self.mass[self.next_dimer_coll.mono_1])/(self.mass[self.next_dimer_coll.mono_2]+self.mass[self.next_dimer_coll.mono_1])) * np.dot(dist_till_coll,speed_diff)* dist_till_coll
    
        
    def snapshot(self, FileName = './snapshot.png', Title = ''):
        '''
        ---> Overwriting snapshot(...) of Monomers class!
        Function saves a snapshot of current configuration,
        i.e. monomer positions as circles of corresponding radius,
        dimer bond length as back empty circles (on top of monomers)
        velocities as arrows on monomers,
        blue dashed lines for the hard walls of the simulation box.
        '''
        fig, ax = plt.subplots( dpi=300 )
        L_xMin, L_xMax = self.BoxLimMin[0], self.BoxLimMax[0]
        L_yMin, L_yMax = self.BoxLimMin[1], self.BoxLimMax[1]
        BorderGap = 0.1*(L_xMax - L_xMin)
        ax.set_xlim(L_xMin-BorderGap, L_xMax+BorderGap)
        ax.set_ylim(L_yMin-BorderGap, L_yMax+BorderGap)

        #--->plot hard walls (rectangle)
        rect = mpatches.Rectangle((L_xMin,L_yMin), L_xMax-L_xMin, L_yMax-L_yMin, linestyle='dashed', ec='gray', fc='None')
        ax.add_patch(rect)
        ax.set_aspect('equal')
        ax.set_xlabel('$x$ position')
        ax.set_ylabel('$y$ position')
        
        #--->plot monomer positions as circles
        COLORS = np.linspace(0.2,0.95,self.ND+1)
        MonomerColors = np.ones(self.NM)*COLORS[-1] #unique color for monomers
        # recolor each monomer pair with individual color
        MonomerColors[self.dimer_pairs[:,0]] = COLORS[:len(self.dimer_pairs)]
        MonomerColors[self.dimer_pairs[:,1]] = COLORS[:len(self.dimer_pairs)]

        #plot solid monomers
        Width, Hight, Angle = 2*self.rad, 2*self.rad, np.zeros( self.NM )
        collection = EllipseCollection( Width, Hight, Angle, units='x', offsets=self.pos,
                       transOffset=ax.transData, cmap='nipy_spectral', edgecolor = 'k')
        collection.set_array(MonomerColors)
        collection.set_clim(0, 1) # <--- we set the limit for the color code
        ax.add_collection(collection)
        
        #plot bond length of dimers as black cicles
        Width, Hight, Angle = self.bond_length, self.bond_length, np.zeros( self.ND )
        mono_i = self.dimer_pairs[:,0]
        mono_j = self.dimer_pairs[:,1]
        collection_mono_i = EllipseCollection( Width, Hight, Angle, units='x', offsets=self.pos[mono_i],
                       transOffset=ax.transData, edgecolor = 'k', facecolor = 'None')
        collection_mono_j = EllipseCollection( Width, Hight, Angle, units='x', offsets=self.pos[mono_j],
                       transOffset=ax.transData, edgecolor = 'k', facecolor = 'None')
        ax.add_collection(collection_mono_i)
        ax.add_collection(collection_mono_j)

        #--->plot velocities as arrows
        ax.quiver( self.pos[:,0], self.pos[:,1], self.vel[:,0], self.vel[:,1] , units = 'dots', scale_units = 'dots')
        
        plt.title(Title)
        plt.savefig( FileName)
        plt.close()
