package io.improbable.keanu.algorithms.mcmc.proposal;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.algorithms.Variable;
import io.improbable.keanu.vertices.VertexId;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.Mock;
import org.mockito.junit.MockitoJUnitRunner;

import java.util.List;
import java.util.Set;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoMoreInteractions;
import static org.mockito.Mockito.when;

@RunWith(MockitoJUnitRunner.class)
public class PriorProposalDistributionTest {

    @Mock
    public GaussianVertex vertex1;
    @Mock
    public GaussianVertex vertex2;
    private PriorProposalDistribution proposalDistribution;

    @Before
    public void setUpProposalDistribution() throws Exception {
        when(vertex1.getReference()).thenReturn(new VertexId(1));
        when(vertex2.getReference()).thenReturn(new VertexId(2));
        proposalDistribution = new PriorProposalDistribution(ImmutableList.of(vertex1, vertex2));
    }

    @Before
    public void setRandomSeed() throws Exception {
        KeanuRandom.setDefaultRandomSeed(0);
    }

    @Test
    public void youCanAddProposalListeners() {
        ProposalListener listener1 = mock(ProposalListener.class);
        ProposalListener listener2 = mock(ProposalListener.class);
        List<ProposalListener> listeners = ImmutableList.of(listener1, listener2);
        proposalDistribution = new PriorProposalDistribution(ImmutableList.of(vertex1, vertex2), listeners);
        Set<Variable> variables = ImmutableSet.of(vertex1, vertex2);
        Proposal proposal = proposalDistribution.getProposal(variables, KeanuRandom.getDefaultRandom());
        verify(listener1).onProposalCreated(proposal);
        verify(listener2).onProposalCreated(proposal);
        proposalDistribution.onProposalRejected();
        verify(listener1).onProposalRejected(proposal);
        verify(listener2).onProposalRejected(proposal);
        verifyNoMoreInteractions(listener1, listener2);
    }
}
