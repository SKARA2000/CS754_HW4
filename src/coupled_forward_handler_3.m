classdef coupled_forward_handler_3
    properties
        transform_handler
        tomo_transform_handler
        measurement_size
        original_size
        angles_1
        angles_2
        angles_3
    end
    
    methods
        function obj = coupled_forward_handler_3(u, tomo_transform, measurement_size, ...
                original_size, angles_1, angles_2, angles_3)
            obj.transform_handler = u;
            obj.tomo_transform_handler = tomo_transform;
            obj.measurement_size = measurement_size;
            obj.original_size = original_size;
            obj.angles_1 = angles_1;
            obj.angles_2 = angles_2;
            obj.angles_3 = angles_3;
        end        
        function output = mtimes(A, X)
            lenX = length(X);
            % Separating the coupled Beta vector into three separate vectors
            % and computing U*Beta for each vector
            temp = X(1:lenX/3);
            delta_temp1 = X(lenX/3 + 1: 2*lenX/3);
            delta_temp2 = X(2*lenX/3 + 1: end);
            temp = reshape(temp, A.original_size, A.original_size);
            delta_temp1 = reshape(delta_temp1, A.original_size, A.original_size);
            delta_temp2  = reshape(delta_temp2, A.original_size, A.original_size);
            
            UX = A.transform_handler(temp);
            delta_UX1 = A.transform_handler(delta_temp1);
            delta_UX2 = A.transform_handler(delta_temp2);
            % We now compute the merged matrix by multiplying individual
            % parts like R1*U*Beta and so on.            
            R1 = A.tomo_transform_handler(UX, A.angles_1);
            R2 = A.tomo_transform_handler(UX, A.angles_2);
            R3 = A.tomo_transform_handler(UX, A.angles_3);
            R1_2 = A.tomo_transform_handler(delta_UX1, A.angles_2);
            R1_3 = A.tomo_transform_handler(delta_UX2, A.angles_3);
            output = [R1(:); R2(:) + R1_2(:); R3(:) + R1_3(:)];     % Vectorizing the result
        end
    end
end