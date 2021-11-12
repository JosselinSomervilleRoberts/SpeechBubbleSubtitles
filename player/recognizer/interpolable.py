from math import factorial



class Interpolable:

    MAX_INTERPOLATION_INTERVAL = 9
    
    def __init__(self):
        self.values = [] # floats
        self.indexes = [] # ints
       
        
       
    def add(self, value, frame_index):
        self.values.append(value)
        self.indexes.append(frame_index)
        
        
        
    def get(self, frame_index, average_nb=9):
        
        # S'il n'y a pas de valeurs
        if len(self.values) == 0:
            print("WARNING: no value available for prediction in Interpolable.get()")
            return 0
        
        
        # On cherche à prédir le futur
        if frame_index > self.indexes[-1]: 
            # On utilise un développement de Taylor
            # Ici on ira pas plus loin que l'ordre 2 (s'il est disponible)
            dframe = frame_index - self.indexes[-1]
            
            # Ordre 0
            value = self.values[-1]
            
            # Ordre 1
            if len(self.values) >= 2:
                deriv1_1 = (self.values[-1] - self.values[-2])/ (self.indexes[-1] - self.indexes[-2])
                value += deriv1_1 * dframe 
            
            # Ordre 2
            if len(self.values) >= 3:
                deriv2_1 = (self.values[-2] - self.values[-3])/ (self.indexes[-2] - self.indexes[-3])
                deriv1_2 = (deriv1_1 - deriv2_1)/ (self.indexes[-1] - self.indexes[-2])
                value += deriv1_2 * dframe**2 / 2.
            
            return value
        
        
        # On cherche une valeur dans le passé (ce qui ne devrait pas arriver)
        if frame_index < self.indexes[0]:
            # On utilise un développement de Taylor
            # Ici on ira pas plus loin que l'ordre 2 (s'il est disponible)
            dframe = frame_index - self.indexes[0]
            
            # Ordre 0
            value = self.values[0]
            
            # Ordre 1
            if len(self.values) >= 2:
                deriv1_1 = (self.values[1] - self.values[0])/ (self.indexes[1] - self.indexes[0])
                value += deriv1_1 * dframe 
            
            # Ordre 2
            if len(self.values) >= 3:
                deriv2_1 = (self.values[2] - self.values[1])/ (self.indexes[2] - self.indexes[1])
                deriv1_2 = (deriv2_1 - deriv1_1)/ (self.indexes[1] - self.indexes[0])
                value += deriv1_2 * dframe**2 / 2.
            
            return value
        
        
        # Si on arrive ici c'est que  self.indexes[0] <= frame_index <= self.indexes[-1]
        # donc soit l'index est dans la liste et on a la valeur exacte, soit on peut interpoler (ici linéairement)
        # On cherche l'index le plus proche en dessous de frame_index
        value_index = 0
        while (len(self.indexes) > value_index +1) and (self.indexes[value_index+1] <= frame_index):
            value_index += 1
            
        # Si on a la valeur exacte
        if self.indexes[value_index] == frame_index:
            demi_fenetre = int((average_nb -1) / 2.)

            sum_coef = 0
            sum_value = 0
            for index in range(max(0, value_index - demi_fenetre), min(len(self.indexes), value_index + demi_fenetre + 1)):
                coef = 1. / (1. + abs(self.indexes[index] - frame_index))
                sum_coef  += coef
                sum_value += coef * self.values[index]

            value = sum_value / sum_coef
            return value # On renvoie la moyenne

        else: # On interpole
            if len(self.indexes) > value_index + 1: # On vérifie mais normalement inutile
                
                if self.indexes[value_index+1] - self.indexes[value_index] > Interpolable.MAX_INTERPOLATION_INTERVAL: # This should not happen, but because of a bug it does
                    return -1
                else:
                    return self.values[value_index] + (self.values[value_index+1] - self.values[value_index]) * (frame_index - self.indexes[value_index]) / float(self.indexes[value_index+1] - self.indexes[value_index])
            else: # si on retrouve ici il y a un probleme mais ons ait jamais
                print("WARNING: no available value for interpolation in Interpolable.get()")
                return self.values[value_index] # On renvoie la valuer la plus proche
            
            
            
    def cleanup(self, frame_index):
        """Delete all the information on the frame previous to frame_index
        We keep at least 4 values previous to frame_index for interpolation purposes
        At the end, we have the first 3 indexes <= frame_index and the rest > frame_index"""
        self.mouth.cleanup(frame_index)
        while  (len(self.indexes) >= 4) and self.indexes[3] <= frame_index:
             self.indexes.pop(0)
             self.values.pop(0)


    def merge(self, other):
        self.indexes.extend(other.indexes)
        self.values.extend(other.values)