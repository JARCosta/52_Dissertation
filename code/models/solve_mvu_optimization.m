function K = solve_mvu_optimization(X, inner_prod, NM, eps, mode)
    % SOLVE_MVU_OPTIMIZATION Solves the Maximum Variance Unfolding optimization problem
    % Inputs:
    %   X - Original data matrix
    %   inner_prod - Inner product matrix of original data
    %   NM - Neighborhood matrix
    %   eps - Tolerance for optimization
    %   mode - 0 for equality constraints, 1 for inequality constraints
    
    % Convert inputs to double if they aren't already
    X = double(X);
    inner_prod = double(inner_prod);
    NM = double(NM);
    eps = double(eps);
    mode = double(mode);
    
    n_samples = size(X, 1);

    % Compute inner product difference matrix
    inner_prod_diff = inner_prod;
    inner_prod_diff(1:n_samples+1:end) = 0;  % Set diagonal to 0
    inner_prod_diff = inner_prod_diff - 2 * inner_prod + inner_prod';

    % Get non-zero elements of neighborhood matrix
    [i, j] = find(NM);
    n_edges = length(i);

    % Pre-compute constraint values
    constraint_vals = inner_prod_diff(sub2ind(size(inner_prod_diff), i, j));

    disp('Variables set.')
    
    % Use CVX to solve the optimization problem
    cvx_begin sdp
        % Create optimization variable
        variable K(n_samples, n_samples) symmetric
        
        % Objective function: maximize trace(K)
        maximize(trace(K))
        
        % Centering constraint
        sum(sum(K)) == 0
        
        % PSD constraint
        K >= 0
        
        % Distance-preserving constraints
        if mode == 0
            % Equality constraints - vectorized
            constraint_vals == K(sub2ind(size(K), i, i)) - 2*K(sub2ind(size(K), i, j)) + K(sub2ind(size(K), j, j))
        else
            % Inequality constraints - vectorized
            constraint_vals >= K(sub2ind(size(K), i, i)) - 2*K(sub2ind(size(K), i, j)) + K(sub2ind(size(K), j, j))
        end
    cvx_end
    
    % Check if the problem was solved successfully
    if strcmp(cvx_status, 'Solved')
        % Problem solved successfully
        return
    % elseif strcmp(cvx_status, 'Infeasible') || strcmp(cvx_status, 'Unbounded')
    %     % Problem is infeasible or unbounded
    %     error('MVU optimization failed: Problem is %s', cvx_status);
    else
        % Any other status is an error
        warning('MVU optimization failed with unexpected status: %s', cvx_status);
    end
end 