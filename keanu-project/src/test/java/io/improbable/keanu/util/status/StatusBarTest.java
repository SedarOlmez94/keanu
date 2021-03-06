package io.improbable.keanu.util.status;

import io.improbable.keanu.testcategory.Slow;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.experimental.categories.Category;
import org.mockito.invocation.InvocationOnMock;
import org.mockito.stubbing.Answer;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.io.UnsupportedEncodingException;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.atomic.AtomicReference;

import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.not;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.atLeastOnce;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoMoreInteractions;
import static org.mockito.Mockito.when;

public class StatusBarTest {

    private AtomicReference<Runnable> progressUpdateCall;
    private StatusBar statusBar;
    private ByteArrayOutputStream byteArrayOutputStream;
    private ScheduledExecutorService scheduler;

    @Before
    public void setup() throws UnsupportedEncodingException {

        byteArrayOutputStream = new ByteArrayOutputStream();
        PrintStream printStream = new PrintStream(byteArrayOutputStream, true, "UTF-8");
        scheduler = mock(ScheduledExecutorService.class);

        progressUpdateCall = new AtomicReference<>(null);

        when(scheduler.scheduleAtFixedRate(any(), anyLong(), anyLong(), any()))
            .thenAnswer(invocation -> {
                progressUpdateCall.set(invocation.getArgument(0));
                return null;
            });

        statusBar = new StatusBar(printStream, scheduler);
        StatusBar.enable();
    }

    @After
    public void cleanup() throws IOException {
        byteArrayOutputStream.close();
    }

    private void convertCrToNewLine(byte[] outputBytes) {
        for (int i = 0; i < outputBytes.length; i++) {
            if (outputBytes[i] == '\r') {
                outputBytes[i] = '\n';
            }
        }
    }

    private String getResultWithNewLinesInsteadOfCR() {
        byte[] outputBytes = byteArrayOutputStream.toByteArray();
        convertCrToNewLine(outputBytes);
        return new String(outputBytes, StandardCharsets.UTF_8);
    }

    @Test
    public void doesPrintToStreamWhenEnabled() {
        StatusBar.enable();

        progressUpdateCall.get().run();
        progressUpdateCall.get().run();
        progressUpdateCall.get().run();
        statusBar.finish();

        String result = getResultWithNewLinesInsteadOfCR();

        assertEquals(5, result.split("\n").length);
    }

    @Test
    public void doesNotPrintToStreamWhenGloballyDisabled() {
        StatusBar.disable();

        progressUpdateCall.get().run();
        progressUpdateCall.get().run();
        statusBar.finish();

        String result = getResultWithNewLinesInsteadOfCR();

        assertEquals("", result);
    }

    @Test
    public void doesCallFinishHandler() {
        StatusBar.enable();

        Runnable finishHandler = mock(Runnable.class);
        statusBar.addFinishHandler(finishHandler);
        progressUpdateCall.get().run();
        statusBar.finish();

        verify(finishHandler).run();
        verifyNoMoreInteractions(finishHandler);
    }

    @Category(Slow.class)
    @Test
    public void youCanOverrideTheDefaultPrintStream() {
        PrintStream mockStream = mock(PrintStream.class);
        doAnswer(new Answer() {
            @Override
            public Object answer(InvocationOnMock invocation) throws Throwable {
                System.out.println(invocation.getArgument(0).toString());
                return null;
            }
        }).when(mockStream).print(anyString());

        StatusBar.setDefaultPrintStream(mockStream);
        StatusBar statusBar = new StatusBar(scheduler);
        StatusBar.enable();
        statusBar.finish();
        verify(mockStream, atLeastOnce()).print(anyString());
    }

    @Test
    public void addedComponentIsRendered() {
        StatusBarComponent mockComponent = mock(StatusBarComponent.class);
        when(mockComponent.render()).thenReturn("RenderTest");

        statusBar.addComponent(mockComponent);
        progressUpdateCall.get().run();
        statusBar.finish();

        String result = getResultWithNewLinesInsteadOfCR();
        assertThat(result, containsString("RenderTest"));
    }

    @Test
    public void removedComponentIsNotRendered() {
        StatusBarComponent mockComponent = mock(StatusBarComponent.class);
        when(mockComponent.render()).thenReturn("RenderTest");

        statusBar.addComponent(mockComponent);
        statusBar.removeComponent(mockComponent);
        progressUpdateCall.get().run();
        statusBar.finish();

        String result = getResultWithNewLinesInsteadOfCR();
        assertThat(result, not(containsString("RenderTest")));
    }

    @Test
    public void setMessageDisplaysMessage() {
        statusBar.setMessage("setMessageTest");
        progressUpdateCall.get().run();
        statusBar.finish();

        String result = getResultWithNewLinesInsteadOfCR();
        assertThat(result, containsString("setMessageTest"));
    }

    @After
    public void tearDown() throws Exception {
        StatusBar.setDefaultPrintStream(System.out);
        StatusBar.disable();
    }
}
