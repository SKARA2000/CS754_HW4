classdef coupled_forward_handler
    properties
        transform_handler
        tomo_transform_handler
        measurement_size
        original_size
        angles_1
        angles_2
    end
    
    methods
        function obj = coupled_forward_handler(u, tomo_transform, measurement_size, ...
                original_size, angles_1, angles_2)
            obj.transform_handler = u;
            obj.tomo_transform_handler = tomo_transform;
            obj.measurement_size = measurement_size;
            obj.original_size = original_size;
            obj.angles_1 = angles_1;
            obj.angles_2 = angles_2;
        end        
        function output = mtimes(A, X)
            lenX = length(X);
            % Separating the coupled Beta vector into two separate vectors
            % and computing U*Beta for each vector
            temp = X(1:lenX/2);
            delta_temp = X(0.5*lenX + 1: end);
            temp = reshape(temp, A.original_size, A.original_size);
            delta_temp = reshape(delta_temp, A.original_size, A.original_size);
            Beta = A.transform_handler(temp);
            delta_Beta = A.transform_handler(delta_temp);
            % We now compute the merged matrix by multiplying individual
            % parts like R1*U*Beta and so on.
            R1 = A.tomo_transform_handler(Beta, A.angles_1);
            R2 = A.tomo_transform_handler(Beta, A.angles_2);
            R2_delta = A.tomo_transform_handler(delta_Beta, A.angles_2);        
            output = [R1(:); R2(:) + R2_delta;              % Vectorizing the result
        end
    end
end