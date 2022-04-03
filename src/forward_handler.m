classdef forward_handler
    properties
        transform_handler
        tomo_transform_handler
        measurement_size
        original_size
        angles
    end
    
    methods
        function obj = forward_handler(u, tomo_transform, measurement_size, ...
                original_size, angles)
            obj.transform_handler = u;
            obj.tomo_transform_handler = tomo_transform;
            obj.measurement_size = measurement_size;
            obj.original_size = original_size;
            obj.angles = angles;
        end
        function output = mtimes(A, X)
            X = reshape(X, A.original_size, A.original_size);               % Reshaping the beta vector
            UX = A.transform_handler(X);                                    % computing U*Beta
            output = A.tomo_transform_handler(UX, A.angles);                % computing R*U*Beta
            output = output(:);                                             % Vectorizing the result
        end
    end
end