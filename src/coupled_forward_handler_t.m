classdef coupled_forward_handler_t
    properties
        transform_handler
        tomo_transform_handler
        measurement_size
        original_size
        angles_1
        angles_2
    end
    
    methods
        function obj = coupled_forward_handler_t(u, tomo_transform, measurement_size, ...
                original_size, angles_1, angles_2)
            obj.transform_handler = u;
            obj.tomo_transform_handler = tomo_transform;
            obj.measurement_size = measurement_size;
            obj.original_size = original_size;
            obj.angles_1 = angles_1;
            obj.angles_2 = angles_2;
        end        
        function output = mtimes(At, Y)
            lenY = length(Y);
            % Separating the coupled Y vector into two separate vectors
            % and computing R'*Beta for each vector            
            Y_1 = Y(1:lenY/2);
            Y_2 = Y(0.5*lenY + 1: end);
            Y_1 = reshape(Y_1, At.measurement_size, size(At.angles_1, 2));
            Y_2 = reshape(Y_2, At.measurement_size, size(At.angles_2, 2));
            Beta = At.tomo_transform_handler(Y_1, At.angles_1, 'linear', 'Ram-Lak', 1, At.original_size);
            delta_Beta = At.tomo_transform_handler(Y_2, At.angles_2, 'linear', 'Ram-Lak', 1, At.original_size);
            % We now compute the merged Beta matrix by computing for Beta
            % and delta_Beta1 which we can later individually sum up to get
            % Beta1 and Beta2 respectively.
            X = At.transform_handler(Beta);
            delta_X = At.transform_handler(delta_Beta);
            output = [X(:) + delta_X(:); delta_X(:)];   % Vectorizing the result
        end
    end
end