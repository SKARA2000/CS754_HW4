classdef forward_handler_t
    properties
        transform_handler
        tomo_transform_handler
        measurement_size
        original_size
        angles
    end
    
    methods
        function obj = forward_handler_t(u, tomo_transform, measurement_size, ...
                original_size, angles)
            obj.transform_handler = u;
            obj.tomo_transform_handler = tomo_transform;
            obj.measurement_size = measurement_size;
            obj.original_size = original_size;
            obj.angles = angles;
        end
        function output = mtimes(At, Y)
            Y = reshape(Y, At.measurement_size, size(At.angles, 2));        % Reshaping the Y vector
            UBeta = At.tomo_transform_handler(Y, At.angles, 'linear', 'Ram-Lak', 1, At.original_size);  % computing R'*Beta
            output = At.transform_handler(UBeta);                           % computing U'*R'*Beta
            output = output(:);                                             % Vectorizing the result
        end
    end
end