function [G, cvx_status] = solve_mvu_optimization(X, N, eps)
    % SOLVE_MVU_OPTIMIZATION Solves the Maximum Variance Unfolding optimization problem
    % Inputs:
    % X: n-by-D data matrix (rows are points)
    % N: set of index pairs (i,j) in the neighborhood (e.g., from k-NN)

    n = size(X, 1);

    inner_prod = X * X';
    % ratio = round(log10(max(inner_prod(:)))) - 2;
    % ratio = 10^(ratio);
    % disp(ratio)
    % inner_prod = inner_prod * ratio;
    
    % D = pdist2(X, X).^2; % squared pairwise distances
    % disp(eps)

    % ==== Step 3: Solve MVU via CVX ====
    cvx_begin sdp
        cvx_solver mosek

        % cvx_solver_settings('MSK_DPAR_INTPNT_CO_TOL_PFEAS', eps)
        % cvx_solver_settings('MSK_DPAR_INTPNT_CO_TOL_DFEAS', eps)
        % cvx_solver_settings('MSK_DPAR_INTPNT_CO_TOL_NEAR_REL', 1/eps)

        variable G(n, n) symmetric
        maximize( trace(G) )
        subject to
            G >= 0;
            sum(G(:)) == 0;
            % trace(G) <= n;
            % norm(G, 'fro') <= sqrt(n);

            % Extract indices for vectorized operations
            i_indices = N(:, 1);
            j_indices = N(:, 2);
            
            G_diag = diag(G);
            gram_distances = G_diag(i_indices) + G_diag(j_indices) - 2*G(sub2ind([n,n], i_indices, j_indices));
            
            inner_prod_diag = diag(inner_prod);
            distances = inner_prod_diag(i_indices) + inner_prod_diag(j_indices) - 2*inner_prod(sub2ind([n,n], i_indices, j_indices));
            gram_distances == distances;
            % gram_distances == D(sub2ind([n,n], i_indices, j_indices));
        
            cvx_end
        disp(trace(G))
    % if ratio ~= 1
    %     G = G / ratio;
    % end
end 