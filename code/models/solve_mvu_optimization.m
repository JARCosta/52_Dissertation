function [G, cvx_status] = solve_mvu_optimization(X, N)
    % SOLVE_MVU_OPTIMIZATION Solves the Maximum Variance Unfolding optimization problem
    % Inputs:
    % X: n-by-D data matrix (rows are points)
    % N: set of index pairs (i,j) in the neighborhood (e.g., from k-NN)

    n = size(X, 1);
    % Compute squared distance matrix without pdist2
    % D(i,j) = ||x_i - x_j||^2 = sum((x_i - x_j).^2)
    D = zeros(n, n);
    for i = 1:n
        for j = 1:n
            D(i,j) = sum((X(i,:) - X(j,:)).^2);
        end
    end

    eps = 1e-3;
    max_iter = 1000;

    % ==== Step 3: Solve MVU via CVX ====
    cvx_begin sdp
        cvx_solver mosek
        % cvx_precision low
        cvx_solver_settings('MSK_DPAR_INTPNT_CO_TOL_PFEAS', eps, ...
                        'MSK_DPAR_INTPNT_CO_TOL_DFEAS', eps, ...
                        'MSK_DPAR_INTPNT_CO_TOL_REL_GAP', eps, ...
                        'MSK_IPAR_INTPNT_MAX_ITERATIONS', max_iter)
        
        variable G(n, n) symmetric
        maximize( trace(G) )
        subject to
            G >= 0;
            sum(sum(G)) == 0;  % centering constraint

            % Vectorized distance-preserving constraints for neighbors
            % Extract indices for vectorized operations
            i_indices = N(:, 1);
            j_indices = N(:, 2);
            
            % Create vectorized constraint: G(i,i) + G(j,j) - 2*G(i,j) == D(i,j) for all pairs
            G_diag = diag(G);
            G_diag(i_indices) + G_diag(j_indices) - 2*G(sub2ind([n,n], i_indices, j_indices)) <= D(sub2ind([n,n], i_indices, j_indices));
        cvx_end
end 